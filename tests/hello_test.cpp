#include <gtest/gtest.h>

#include "ttml.hpp"
// Demonstrate some basic assertions.
TEST(HelloTest, AddTest) {
    // Expect two strings not to be equal.
    // Expect equality.
    EXPECT_EQ(ttml::sum(7, 6), 13);
}