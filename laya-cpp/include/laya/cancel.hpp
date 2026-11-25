#pragma once
// A cancellation token. Copies share one flag, so the caller keeps a copy and cancels it from any
// thread while a call holds another. Every token, default included, owns a flag.
#include <atomic>
#include <memory>

namespace laya {

class CancelToken {
 public:
  CancelToken() : flag_(std::make_shared<std::atomic<bool>>(false)) {}
  static CancelToken Make() { return CancelToken(); }
  void Cancel() const { flag_->store(true, std::memory_order_release); }
  bool cancelled() const { return flag_->load(std::memory_order_acquire); }

 private:
  std::shared_ptr<std::atomic<bool>> flag_;
};

}  // namespace laya
