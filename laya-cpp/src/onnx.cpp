#include "laya/onnx.hpp"

#include <onnxruntime_cxx_api.h>

#include <filesystem>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <set>
#include <thread>

#include "laya/onnx_graph.hpp"

namespace laya {
namespace onnx {
namespace {

// ONNX Runtime 1.30 on POSIX starts a telemetry uploader thread with its first OrtEnv unless
// ORT_DISABLE_TELEMETRY is truthy then. At exit that thread can lock a mutex the runtime's statics
// already destroyed, which aborts the process ("recursive_mutex lock failed"). The variable is set
// before our Env is built; a value the user already set wins.
bool DisableRuntimeTelemetry() {
#ifndef _WIN32
  setenv("ORT_DISABLE_TELEMETRY", "1", 0);
#endif
  return true;
}

// One Ort::Env per process, built on the first Open. The logging level is FATAL so a refused run is
// reported only through its Result.
Ort::Env& SharedEnv() {
  static const bool telemetry_off = DisableRuntimeTelemetry();
  static Ort::Env env(ORT_LOGGING_LEVEL_FATAL, "laya");
  (void)telemetry_off;
  return env;
}

std::string Dtype(ONNXTensorElementDataType t) {
  switch (t) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return "int64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return "bool";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return "float32";
    default: return "type " + std::to_string(int(t));
  }
}

// Inputs by name, in the order the five tensors are bound below.
const char* const kInputNames[5] = {"input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"};
const char* const kOutputNames[2] = {"logits", "act_logits"};

// Fires SetTerminate on a run when the cancel token fires. The token is a polled flag, so the
// watcher waits on a condition variable with a short timeout; the run ending wakes it at once.
class Watcher {
 public:
  Watcher(const CancelToken& cancel, Ort::RunOptions& ro) : cancel_(cancel), ro_(ro), thread_([this] { Loop(); }) {}
  ~Watcher() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      done_ = true;
    }
    cv_.notify_all();
    thread_.join();
  }
  bool terminated() const { return terminated_.load(); }

 private:
  void Loop() {
    std::unique_lock<std::mutex> lock(mu_);
    while (!done_) {
      if (cancel_.cancelled()) {
        terminated_ = true;
        ro_.SetTerminate();
        return;
      }
      cv_.wait_for(lock, std::chrono::milliseconds(2));
    }
  }
  CancelToken cancel_;
  Ort::RunOptions& ro_;
  std::mutex mu_;
  std::condition_variable cv_;
  bool done_ = false;
  std::atomic<bool> terminated_{false};
  std::thread thread_;  // last: started after the rest is built
};

class OrtSession final : public infer::Session {
 public:
  explicit OrtSession(std::unique_ptr<Ort::Session> s) : session_(std::move(s)) {}
  ~OrtSession() override { Close(); }

  Result<infer::Output> Run(const infer::Batch& b, const CancelToken& cancel) override {
    try {
      return RunImpl(b, cancel);
    } catch (const Ort::Exception& e) {
      return Result<infer::Output>::Fail(std::string("onnx: run: ") + e.what());
    } catch (const std::exception& e) {
      return Result<infer::Output>::Fail(std::string("onnx: run: ") + e.what());
    } catch (...) {
      return Result<infer::Output>::Fail("onnx: run: unknown exception");
    }
  }

  // Refuses new runs, terminates the runs in flight, and releases the session once they have left.
  std::string Close() override {
    try {
      std::unique_lock<std::mutex> lock(mu_);
      closing_ = true;
      for (Ort::RunOptions* r : flights_) r->SetTerminate();
      idle_.wait(lock, [this] { return flights_.empty(); });
      session_.reset();
      return "";
    } catch (...) {
      return "onnx: close failed";
    }
  }

 private:
  // Registers a run for Close to terminate; the destructor leaves the list and wakes Close.
  class Flight {
   public:
    Flight(OrtSession& s, Ort::RunOptions& ro) : s_(s), ro_(ro) {
      std::lock_guard<std::mutex> lock(s_.mu_);
      if (s_.closing_ || !s_.session_) return;
      s_.flights_.insert(&ro_);
      ok_ = true;
    }
    ~Flight() {
      if (!ok_) return;
      std::lock_guard<std::mutex> lock(s_.mu_);
      s_.flights_.erase(&ro_);
      s_.idle_.notify_all();
    }
    bool ok() const { return ok_; }

   private:
    OrtSession& s_;
    Ort::RunOptions& ro_;
    bool ok_ = false;
  };

  Result<infer::Output> RunImpl(const infer::Batch& b, const CancelToken& cancel) {
    using R = Result<infer::Output>;
    if (cancel.cancelled()) return R::Fail(infer::kCancelled);
    const std::string bad = b.Validate();
    if (!bad.empty()) return R::Fail("onnx: " + bad);
    for (uint8_t m : b.marker_mask) {
      if (m > 1) return R::Fail("onnx: marker_mask holds a value other than 0 or 1");
    }
    Ort::RunOptions ro;
    Flight flight(*this, ro);
    if (!flight.ok()) return R::Fail("onnx: session is closed");

    const Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const int64_t shape2[2] = {b.rows, b.length};
    const int64_t shape_m[2] = {b.rows, b.markers};
    const int64_t shape1[1] = {b.rows};
    // ORT takes non-const pointers but never writes to an input.
    auto i64 = [&](const std::vector<int64_t>& v, const int64_t* shape, size_t rank) {
      return Ort::Value::CreateTensor<int64_t>(mem, const_cast<int64_t*>(v.data()), v.size(), shape, rank);
    };
    std::vector<Ort::Value> in;
    in.reserve(5);
    in.push_back(i64(b.input_ids, shape2, 2));
    in.push_back(i64(b.attention_mask, shape2, 2));
    in.push_back(i64(b.marker_pos, shape_m, 2));
    in.push_back(Ort::Value::CreateTensor(mem, const_cast<uint8_t*>(b.marker_mask.data()), b.marker_mask.size() * sizeof(uint8_t),
                                          shape_m, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL));
    in.push_back(i64(b.qtype, shape1, 1));

    Watcher watcher(cancel, ro);
    std::vector<Ort::Value> out;
    try {
      out = session_->Run(ro, kInputNames, in.data(), in.size(), kOutputNames, 2);
    } catch (const Ort::Exception& e) {
      if (watcher.terminated() || cancel.cancelled()) return R::Fail(infer::kCancelled);
      throw;
    }
    if (out.size() != 2) return R::Fail("onnx: run returned " + std::to_string(out.size()) + " outputs, want 2");
    if (cancel.cancelled()) return R::Fail(infer::kCancelled);

    OutTensor t[2];
    for (size_t i = 0; i < 2; i++) {
      if (!out[i].IsTensor()) return R::Fail(std::string("onnx: output ") + kOutputNames[i] + " is not a tensor");
      const Ort::TensorTypeAndShapeInfo info = out[i].GetTensorTypeAndShapeInfo();
      if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        return R::Fail(std::string("onnx: output ") + kOutputNames[i] + " is " + Dtype(info.GetElementType()) + ", want float32");
      }
      t[i].shape = info.GetShape();
      t[i].count = int64_t(info.GetElementCount());
      t[i].data = out[i].GetTensorData<float>();
    }
    return CollectOutput(b, t[0], t[1]);
  }

  std::mutex mu_;  // guards closing_, flights_ and the release of session_
  std::condition_variable idle_;
  bool closing_ = false;
  std::set<Ort::RunOptions*> flights_;
  std::unique_ptr<Ort::Session> session_;
};

}  // namespace

std::string RuntimeVersion() {
  try {
    return Ort::GetVersionString();
  } catch (...) {
    return "";
  }
}

Result<std::unique_ptr<infer::Session>> Open(const std::string& graph_path, const Options& options) {
  using R = Result<std::unique_ptr<infer::Session>>;
  if (options.intra_op_threads < 0 || options.intra_op_threads > kMaxThreads) {
    return R::Fail("onnx: intra_op_threads must be between 0 and " + std::to_string(kMaxThreads));
  }
  const std::string path_problem = CheckPath(graph_path);
  if (!path_problem.empty()) return R::Fail("onnx: " + path_problem);
  try {
    Ort::Env& env = SharedEnv();
    Ort::SessionOptions so;
    if (options.intra_op_threads > 0) so.SetIntraOpNumThreads(options.intra_op_threads);
    const std::filesystem::path p(graph_path);
    auto session = std::make_unique<Ort::Session>(env, p.c_str(), so);

    Ort::AllocatorWithDefaultOptions alloc;
    auto describe = [&](bool inputs) {
      std::vector<TensorInfo> v;
      const size_t n = inputs ? session->GetInputCount() : session->GetOutputCount();
      for (size_t i = 0; i < n; i++) {
        TensorInfo t;
        Ort::AllocatedStringPtr name = inputs ? session->GetInputNameAllocated(i, alloc) : session->GetOutputNameAllocated(i, alloc);
        t.name = name.get();
        Ort::TypeInfo ti = inputs ? session->GetInputTypeInfo(i) : session->GetOutputTypeInfo(i);
        if (ti.GetONNXType() != ONNX_TYPE_TENSOR) {
          t.dtype = "not a tensor";
        } else {
          const Ort::ConstTensorTypeAndShapeInfo info = ti.GetTensorTypeAndShapeInfo();
          t.dtype = Dtype(info.GetElementType());
          t.dims = info.GetShape();
        }
        v.push_back(std::move(t));
      }
      return v;
    };
    const std::string bad = CheckGraph(describe(true), describe(false));
    if (!bad.empty()) return R::Fail("onnx: graph " + graph_path + ": " + bad);
    return R::Ok(std::unique_ptr<infer::Session>(new OrtSession(std::move(session))));
  } catch (const Ort::Exception& e) {
    return R::Fail("onnx: cannot open " + graph_path + ": " + e.what());
  } catch (const std::exception& e) {
    return R::Fail("onnx: cannot open " + graph_path + ": " + e.what());
  } catch (...) {
    return R::Fail("onnx: cannot open " + graph_path + ": unknown exception");
  }
}

}  // namespace onnx

namespace agent {
Result<std::unique_ptr<Agent>> OpenOnnx(const std::string& checkpoint_dir, const std::string& graph_path, const onnx::Options& options) {
  auto s = onnx::Open(graph_path, options);
  if (!s.ok()) return Result<std::unique_ptr<Agent>>::Fail(s.error());
  return Agent::Open(checkpoint_dir, std::move(s).value());  // closes the session on failure
}
}  // namespace agent

}  // namespace laya
