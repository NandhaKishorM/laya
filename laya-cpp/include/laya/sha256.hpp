#pragma once
// SHA-256 (FIPS 180-4), self-contained. Used for artifact digests, not for secrets.
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace laya {

class Sha256 {
 public:
  Sha256();
  // Returns false, and changes nothing, once Finish has been called.
  bool Update(const void* data, size_t size);
  bool Update(std::string_view s) { return Update(s.data(), s.size()); }
  // Finishes the hash. Calling it again returns the same digest.
  std::array<uint8_t, 32> Finish();

 private:
  void Block(const uint8_t* p);
  uint32_t h_[8];
  uint8_t buf_[64];
  size_t used_ = 0;
  uint64_t total_ = 0;
  bool finished_ = false;
  std::array<uint8_t, 32> digest_{};
};

std::array<uint8_t, 32> Sha256Of(std::string_view data);
// Lowercase hex, as hashlib.sha256(...).hexdigest().
std::string Sha256Hex(std::string_view data);
std::string ToHex(const std::array<uint8_t, 32>& digest);

}  // namespace laya
