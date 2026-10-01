"""Tests for _InferenceGate read-write synchronization gate (#649).

Ensures that:
1. Normal inference allows concurrent readers without mutual exclusion.
2. An OOM fallback (writer) waits for in-flight readers to finish before migrating the model.
3. While fallback is active (writer lock held), incoming requests (readers) are queued.
4. Writers have preference over new readers so OOM fallback does not starve under load.
5. Exceptions inside read or write contexts cleanly release state without deadlocking.
"""
import os
import sys
import threading
import time
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    import torch
except ImportError:
    t = types.ModuleType("torch")
    # laya/common.py annotates `proper_reward` with both at module scope, so importing
    # laya.agent through this stub needs them present even though no test touches a tensor.
    t.Tensor = type("Tensor", (), {})
    t.dtype = type("dtype", (), {})
    t.cuda = types.ModuleType("torch.cuda")
    t.cuda.OutOfMemoryError = type("OutOfMemoryError", (RuntimeError,), {})
    t.nn = types.ModuleType("torch.nn")
    t.nn.Module = type("Module", (), {})
    t.nn.MultiheadAttention = type("MultiheadAttention", (), {})
    t.nn.functional = types.ModuleType("torch.nn.functional")
    t.utils = types.ModuleType("torch.utils")
    t.utils.checkpoint = types.ModuleType("torch.utils.checkpoint")
    t.utils.checkpoint.checkpoint = lambda f, *a, **kw: f(*a, **kw)
    t.device = lambda d: d
    t.float32 = "float32"
    t.float16 = "float16"
    t.bfloat16 = "bfloat16"
    t.no_grad = lambda: (lambda f: f)
    sys.modules["torch"] = t
    sys.modules["torch.cuda"] = t.cuda
    sys.modules["torch.nn"] = t.nn
    sys.modules["torch.nn.functional"] = t.nn.functional
    sys.modules["torch.utils"] = t.utils
    sys.modules["torch.utils.checkpoint"] = t.utils.checkpoint


from laya.agent import _InferenceGate

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s: got %r, want %r" % (name, got, want))


def test_concurrent_readers():
    gate = _InferenceGate()
    active_readers = []
    max_concurrent = [0]
    lock = threading.Lock()
    all_in = threading.Event()
    entered = 0
    entered_lock = threading.Lock()

    def reader():
        nonlocal entered
        with gate.read_lock():
            with lock:
                active_readers.append(1)
                max_concurrent[0] = max(max_concurrent[0], len(active_readers))
            with entered_lock:
                entered += 1
                if entered == 5:
                    all_in.set()
            all_in.wait(5)
            with lock:
                active_readers.pop()

    threads = [threading.Thread(target=reader) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    check("concurrent_readers/all ran concurrently", max_concurrent[0], 5)
    check("concurrent_readers/all finished", len(active_readers), 0)


def test_writer_mutual_exclusion():
    gate = _InferenceGate()
    events = []
    r_in, r_go = threading.Event(), threading.Event()

    def reader():
        with gate.read_lock():
            events.append("reader_start")
            r_in.set()
            r_go.wait(5)
            events.append("reader_end")

    def writer():
        with gate.write_lock():
            events.append("writer_start")
            events.append("writer_end")

    t_reader = threading.Thread(target=reader)
    t_writer = threading.Thread(target=writer)

    t_reader.start()
    r_in.wait(5)  # ensure reader holds lock
    t_writer.start()  # writer must wait until reader finishes
    while gate._waiting_writers == 0:
        time.sleep(0.001)
    r_go.set()

    t_reader.join()
    t_writer.join()

    check("writer_exclusion/order", events, [
        "reader_start",
        "reader_end",
        "writer_start",
        "writer_end",
    ])


def test_writer_preference():
    gate = _InferenceGate()
    events = []
    r1_in, r1_go = threading.Event(), threading.Event()
    w1_in, w1_go = threading.Event(), threading.Event()

    def reader_1():
        with gate.read_lock():
            events.append("r1_start")
            r1_in.set()
            r1_go.wait(5)
            events.append("r1_end")

    def writer_1():
        with gate.write_lock():
            events.append("w1_start")
            w1_in.set()
            w1_go.wait(5)
            events.append("w1_end")

    def reader_2():
        with gate.read_lock():
            events.append("r2_start")
            events.append("r2_end")

    t_r1 = threading.Thread(target=reader_1)
    t_w1 = threading.Thread(target=writer_1)
    t_r2 = threading.Thread(target=reader_2)

    t_r1.start()
    r1_in.wait(5)  # r1 active and holding read lock
    t_w1.start()
    while gate._waiting_writers == 0:
        time.sleep(0.001)  # w1 registered as waiting writer
    t_r2.start()  # r2 arrives after w1, must wait for w1
    time.sleep(0.02)  # r2 is blocked behind w1
    r1_go.set()  # release r1 so w1 can enter

    w1_in.wait(5)  # w1 is now active
    w1_go.set()  # release w1 so r2 can enter

    t_r1.join()
    t_w1.join()
    t_r2.join()

    check("writer_preference/order", events, [
        "r1_start",
        "r1_end",
        "w1_start",
        "w1_end",
        "r2_start",
        "r2_end",
    ])


def test_exception_safety():
    gate = _InferenceGate()

    try:
        with gate.read_lock():
            raise RuntimeError("reader failed")
    except RuntimeError:
        pass

    check("exception_safety/readers cleared after error", gate._readers, 0)

    try:
        with gate.write_lock():
            raise RuntimeError("writer failed")
    except RuntimeError:
        pass

    check("exception_safety/writers cleared after error", gate._writers, 0)
    check("exception_safety/waiting writers cleared", gate._waiting_writers, 0)

    # Next lock acquisitions should succeed immediately without hanging
    acquired_read = False
    with gate.read_lock():
        acquired_read = True
    check("exception_safety/subsequent read succeeds", acquired_read, True)

    acquired_write = False
    with gate.write_lock():
        acquired_write = True
    check("exception_safety/subsequent write succeeds", acquired_write, True)


def test_multiple_concurrent_writers():
    gate = _InferenceGate()
    active_writers = []
    max_writers = [0]
    lock = threading.Lock()

    def writer(duration):
        with gate.write_lock():
            with lock:
                active_writers.append(1)
                max_writers[0] = max(max_writers[0], len(active_writers))
            time.sleep(duration)
            with lock:
                active_writers.pop()

    threads = [threading.Thread(target=writer, args=(0.02,)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    check("concurrent_writers/strictly serialized", max_writers[0], 1)
    check("concurrent_writers/all finished", len(active_writers), 0)


test_concurrent_readers()
test_writer_mutual_exclusion()
test_writer_preference()
test_exception_safety()
test_multiple_concurrent_writers()

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all agent gate tests passed")
sys.exit(1 if FAIL else 0)
