#include <gtest/gtest.h>

#include <string>

#include "laya/version.hpp"

TEST(Version, IsSemver) {
  const std::string v = laya::version();
  EXPECT_EQ(v, "0.0.0");
}
