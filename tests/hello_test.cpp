#include <gtest/gtest.h>
#include "warp_drive.hpp"
// Demonstrate some basic assertions.
TEST(HelloTest, AddTest) {
    // Expect two strings not to be equal.
    // Expect equality.
    EXPECT_EQ(wd::sum(7, 6), 13);
}