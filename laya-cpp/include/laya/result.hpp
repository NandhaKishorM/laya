#pragma once
// Error handling for the whole library: no exception crosses the public API. A function that can
// fail returns Result<T>; test ok() before value(). value() on a failed Result is a programming
// error and aborts, it never throws.
#include <cstdlib>
#include <string>
#include <utility>

namespace laya {

template <class T>
class Result {
 public:
  static Result Ok(T value) {
    Result r;
    r.value_ = std::move(value);
    r.ok_ = true;
    return r;
  }
  static Result Fail(std::string message) {
    Result r;
    r.error_ = std::move(message);
    return r;
  }
  bool ok() const { return ok_; }
  const std::string& error() const { return error_; }
  const T& value() const& {
    if (!ok_) std::abort();
    return value_;
  }
  T& value() & {
    if (!ok_) std::abort();
    return value_;
  }
  T&& value() && {
    if (!ok_) std::abort();
    return std::move(value_);
  }

 private:
  Result() = default;
  T value_{};
  std::string error_;
  bool ok_ = false;
};

}  // namespace laya
