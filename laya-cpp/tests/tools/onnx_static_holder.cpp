// Holds a Session and an Agent in statics, as a user program may, then exits normally. Exit code 0
// means static destruction did not fault. argv: graph checkpoint_dir [close].
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#include "laya/onnx.hpp"

namespace {
std::unique_ptr<laya::infer::Session> g_session;  // constructed before any ONNX Runtime static
std::unique_ptr<laya::agent::Agent> g_agent;
}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) return 64;
  auto s = laya::onnx::Open(argv[1]);
  if (!s.ok()) {
    std::fprintf(stderr, "open: %s\n", s.error().c_str());
    return 2;
  }
  g_session = std::move(s).value();
  if (std::getenv("ORT_DISABLE_TELEMETRY") == nullptr) return 7;  // Open must opt out first
  auto a = laya::agent::OpenOnnx(argv[2], argv[1]);
  if (!a.ok()) {
    std::fprintf(stderr, "agent: %s\n", a.error().c_str());
    return 3;
  }
  g_agent = std::move(a).value();
  {  // run once so ONNX Runtime's lazily built statics exist, and are built after ours
    laya::infer::Batch b;
    b.rows = 1; b.length = 3; b.markers = 3;
    b.input_ids.assign(3, 1); b.attention_mask.assign(3, 1); b.marker_pos.assign(3, 1);
    b.marker_mask.assign(3, 1); b.qtype.assign(1, 0);
    if (!g_session->Run(b, laya::CancelToken()).ok()) return 5;
    if (!g_agent->Predict(laya::pyvalue::Value::String("hello"), laya::schema::Questions::Parse(
            R"({"q":{"type":"noul","instructions":"x"}})").value()).ok()) return 6;
  }
  {  // concurrent runs right before exit leave the runtime's worker threads busy as statics die
    std::vector<std::thread> ts;
    for (int i = 0; i < 8; i++) ts.emplace_back([] {
      laya::infer::Batch b;
      b.rows = 4; b.length = 64; b.markers = 3;
      b.input_ids.assign(256, 5); b.attention_mask.assign(256, 1); b.marker_pos.assign(12, 1);
      b.marker_mask.assign(12, 1); b.qtype.assign(4, 0);
      for (int k = 0; k < 3; k++) g_session->Run(b, laya::CancelToken());
    });
    for (auto& t : ts) t.join();
  }
  if (argc > 3 && !std::strcmp(argv[3], "close")) {
    if (!g_session->Close().empty() || !g_agent->Close().empty()) return 4;
  }
  return 0;
}
