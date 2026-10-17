#!/usr/bin/env python3
"""The one list of test suites CI and the release gate run.

`ci.yml` used to name the script-style suites inline and `release.yml` kept its own copy of the
list; they drifted, and the release gate ("tests must pass before anything is published") named 33
fewer suites than CI, so a regression in an omitted one could ship to PyPI. Both workflows invoke
this module now, so there is a single list to keep and `tests/test_packaging.py` fails if either
workflow stops using it.

Two kinds of suite, because `python tests/<name>.py` does not execute a pytest suite -- it defines
the test functions and exits 0, having asserted nothing (#374):

* ``SCRIPT_SUITES`` print their own PASS/FAIL summary and exit non-zero on failure.
* ``PYTEST_SUITES`` are collected and run by pytest in a single invocation.

A suite that crashes, rather than failing, still has to leave evidence: children run with
``PYTHONUNBUFFERED`` (their stdout is a pipe here, and a process that dies in native code never
flushes it, so a buffered run loses every line it had already printed) and the failure summary
carries each child's exit status, which is the only thing that tells a crash apart from a failed
assertion.

``tests/test_langchain_real_mode.py`` is in ``SCRIPT_SUITES`` and fails (it does not skip) when
``langchain-core`` is missing, so a lane that runs this list needs the ``langchain`` extra -- which
is why the release build installs the same extras as the CI `test` job.
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCRIPT_SUITES = [
    "tests/test_numeric.py",
    "tests/test_router.py",
    "tests/test_criteria.py",
    "tests/test_batch.py",
    "tests/test_predict_long.py",
    "tests/test_attention_dynamic_shapes.py",
    "tests/test_option_order.py",
    "tests/test_parallel_options.py",
    "tests/test_state_budget.py",
    "tests/test_hooks.py",
    "tests/test_hooks_api.py",
    "tests/test_structured.py",
    "tests/test_structured_api.py",
    "tests/test_structured_docs.py",
    "tests/test_onnx_lang_parity.py",
    "tests/test_onnx_truncation_parity.py",
    "tests/test_onnx_batch.py",
    "tests/test_onnx_long.py",
    "tests/test_onnx_sort.py",
    "tests/test_download.py",
    "tests/test_verify_checkpoints.py",
    "tests/test_shortlist.py",
    "tests/test_decision_model.py",
    "tests/test_head_checkpointing.py",
    "tests/test_truncation.py",
    "tests/test_packaging.py",
    "tests/test_imports.py",
    "tests/test_compose_env.py",
    "tests/test_audit_scope.py",
    "tests/test_doc_tables.py",
    "tests/test_env_docs.py",
    "tests/test_doc_citations.py",
    "tests/test_tokenizer_cache.py",
    "tests/test_tokenizer_concurrency.py",
    "tests/test_question_token_reuse.py",
    "tests/test_lazy_import.py",
    "tests/test_runtime_fixes.py",
    "tests/test_backends.py",
    "tests/test_fast_head_partition.py",
    "tests/test_agent_gate.py",
    "tests/test_option_collapse.py",
    "tests/test_lang_guess.py",
    "tests/test_identifier_complexity.py",
    "tests/test_lang_stats.py",
    "tests/test_calibration_persistence.py",
    "tests/test_calibrate.py",
    "tests/test_context_manager.py",
    "tests/test_criteria_normalization.py",
    "tests/test_docker_entrypoint.py",
    "tests/test_modelscope_prefetch.py",
    "tests/test_empty_questions.py",
    "tests/test_router_memory.py",
    "tests/test_shortlist_cosine.py",
    "tests/test_temperature_loading.py",
    "tests/test_confidence.py",
    "tests/test_conformal_abstention.py",
    "tests/test_cli.py",
    "tests/test_cli_lang_guess.py",
    "tests/test_evidence.py",
    "tests/test_mcp.py",
    "tests/test_mcp_device.py",
    "tests/test_langchain.py",
    "tests/test_portability.py",
    "tests/test_guard_document_781.py",
    "tests/test_training.py",
    "tests/test_train.py",
    "tests/test_kaggle_ddp_convergence.py",
    "tests/test_example_server_limits.py",
    "tests/test_blank_lang_routing.py",
    "tests/test_export_onnx_safety.py",
    "tests/test_load_errors.py",
    "tests/test_revision_pinning.py",
    "tests/test_example_server_errors.py",
    "tests/test_example_server_per_call_controls.py",
    "tests/test_crewai.py",
    "tests/test_llamaindex.py",
    "tests/test_langchain_real_mode.py",
    "tests/test_email.py",
]

PYTEST_SUITES = [
    "tests/test_evals_agreement.py",
    "tests/test_evals_tournament.py",
    "tests/test_serve.py",
    "tests/test_mcp_remote.py",
    "tests/test_router_batch.py",
    "tests/test_predict_batch.py",
    "tests/test_system_one_lang.py",
    "tests/test_audit_regressions.py",
    "tests/test_truncation_direction.py",
    "tests/test_compile.py",
    "tests/test_finetune_entrypoints.py",
]


def suites():
    """Every suite in run order: the scripts, then one pytest invocation."""
    return list(SCRIPT_SUITES) + list(PYTEST_SUITES)


def _commands():
    commands = [[sys.executable, path] for path in SCRIPT_SUITES]
    if PYTEST_SUITES:
        commands.append([sys.executable, "-m", "pytest", *PYTEST_SUITES, "-q"])
    return commands


# A child's stdout here is a pipe, and CPython block-buffers a pipe: a suite that dies in native
# code never flushes it, so the log loses every line it had printed and is left naming a suite
# that said nothing. Unbuffered, the evidence reaches the log before the crash does.
CHILD_ENV = dict(os.environ, PYTHONUNBUFFERED="1")


def _death_note(returncode):
    """How a suite invocation ended, in the one number the child actually left behind.

    A `FAILED`/`ERROR` line says *which* assertion did not hold; a signal (negative, POSIX) or an
    NTSTATUS (Windows) says the suite never reached its own report. Hex goes alongside the decimal
    for those because 3221225477 means nothing while 0xC0000005 is a name.
    """
    if returncode < 0:
        return "killed by signal %d" % -returncode
    if returncode > 0xFF:
        return "exit %d (0x%08X)" % (returncode, returncode)
    return "exit %d" % returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the shared CI/release test suite list.")
    parser.add_argument("--list", action="store_true", help="print the suites and exit")
    args = parser.parse_args(argv)

    if args.list:
        for path in suites():
            print(path)
        return 0

    missing = [p for p in suites() if not os.path.exists(os.path.join(ROOT, p))]
    if missing:
        print("test_suites: suite file(s) missing: %s" % ", ".join(missing))
        return 1

    failed = []
    for command in _commands():
        label = " ".join(command[1:])
        print("\n===== %s =====" % label, flush=True)
        returncode = subprocess.run(command, cwd=ROOT, env=CHILD_ENV).returncode
        if returncode != 0:
            failed.append((label, returncode))

    if failed:
        print("\n%d suite invocation(s) failed:" % len(failed))
        for label, returncode in failed:
            print("  - %s [%s]" % (label, _death_note(returncode)))
        return 1
    print("\nall %d suites passed" % len(suites()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
