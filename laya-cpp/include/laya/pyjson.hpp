#pragma once
// Text the way CPython writes it: repr(float), repr(str), '%.0f' and round(x, 4). These feed the
// tokenizer, so one differing character is a different answer; never use %g, ostream or locale
// formatting for them.
#include <string>
#include <string_view>

namespace laya {
namespace pyjson {

constexpr int kMaxDepth = 512;

// repr(float) as json.dumps writes it: shortest round-trip digits, fixed notation for decimal
// exponents in [-4, 16), else scientific with a signed two-digit exponent. NaN, Infinity,
// -Infinity as json.dumps spells them.
std::string ReprFloat(double v);
// repr(str): single-quoted unless the text holds ' and no ", everything not isprintable escaped.
std::string Repr(std::string_view s);
// '%.0f' % (100 * fraction): halves to even on the exact binary value.
std::string Percent0(double fraction);
// round(x, 4): correctly rounded, ties to even on the exact binary value. NaN and inf pass through.
double Round4(double v);

}  // namespace pyjson
}  // namespace laya
