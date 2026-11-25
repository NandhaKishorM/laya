#include "laya/sha256.hpp"

#include <cstring>

namespace laya {
namespace {

constexpr uint32_t kK[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

inline uint32_t Rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

}  // namespace

Sha256::Sha256()
    : h_{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19},
      buf_{} {}

void Sha256::Block(const uint8_t* p) {
  uint32_t w[64];
  for (int i = 0; i < 16; i++) {  // big-endian words
    w[i] = (uint32_t(p[4 * i]) << 24) | (uint32_t(p[4 * i + 1]) << 16) | (uint32_t(p[4 * i + 2]) << 8) |
           uint32_t(p[4 * i + 3]);
  }
  for (int i = 16; i < 64; i++) {
    uint32_t s0 = Rotr(w[i - 15], 7) ^ Rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
    uint32_t s1 = Rotr(w[i - 2], 17) ^ Rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }
  uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3], e = h_[4], f = h_[5], g = h_[6], h = h_[7];
  for (int i = 0; i < 64; i++) {
    uint32_t t1 = h + (Rotr(e, 6) ^ Rotr(e, 11) ^ Rotr(e, 25)) + ((e & f) ^ (~e & g)) + kK[i] + w[i];
    uint32_t t2 = (Rotr(a, 2) ^ Rotr(a, 13) ^ Rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }
  h_[0] += a;
  h_[1] += b;
  h_[2] += c;
  h_[3] += d;
  h_[4] += e;
  h_[5] += f;
  h_[6] += g;
  h_[7] += h;
}

bool Sha256::Update(const void* data, size_t size) {
  if (finished_) return false;
  const uint8_t* p = static_cast<const uint8_t*>(data);
  total_ += size;
  while (size > 0) {
    size_t take = 64 - used_;
    if (take > size) take = size;
    std::memcpy(buf_ + used_, p, take);
    used_ += take;
    p += take;
    size -= take;
    if (used_ == 64) {
      Block(buf_);
      used_ = 0;
    }
  }
  return true;
}

std::array<uint8_t, 32> Sha256::Finish() {
  if (finished_) return digest_;
  const uint64_t bits = total_ * 8;
  uint8_t pad[72] = {0x80};
  // Pad with 0x80 then zeros to 56 mod 64, then the 64-bit big-endian bit length.
  size_t padlen = (used_ < 56) ? (56 - used_) : (120 - used_);
  uint8_t len[8];
  for (int i = 0; i < 8; i++) len[i] = uint8_t(bits >> (56 - 8 * i));
  Update(pad, padlen);
  Update(len, 8);
  finished_ = true;
  std::array<uint8_t, 32>& out = digest_;
  for (int i = 0; i < 8; i++) {
    out[4 * i] = uint8_t(h_[i] >> 24);
    out[4 * i + 1] = uint8_t(h_[i] >> 16);
    out[4 * i + 2] = uint8_t(h_[i] >> 8);
    out[4 * i + 3] = uint8_t(h_[i]);
  }
  return out;
}

std::array<uint8_t, 32> Sha256Of(std::string_view data) {
  Sha256 s;
  s.Update(data);
  return s.Finish();
}

std::string ToHex(const std::array<uint8_t, 32>& digest) {
  static const char* kHex = "0123456789abcdef";
  std::string out;
  out.reserve(64);
  for (uint8_t b : digest) {
    out.push_back(kHex[b >> 4]);
    out.push_back(kHex[b & 15]);
  }
  return out;
}

std::string Sha256Hex(std::string_view data) { return ToHex(Sha256Of(data)); }

}  // namespace laya
