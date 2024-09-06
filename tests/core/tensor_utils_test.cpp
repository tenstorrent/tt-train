

#include "core/tensor_utils.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "core/device.hpp"
#include "core/ttnn_all_includes.hpp"

class TensorUtilsTest : public ::testing::Test {
protected:
    void SetUp() override { device = std::make_unique<ttml::core::Device>(0); }

    void TearDown() override { device.reset(); }

    std::unique_ptr<ttml::core::Device> device;
};

TEST_F(TensorUtilsTest, TestToFromTensor) {
    std::vector<float> test_data = {1.F, 5.F, 10.F, 15.F};

    auto tensor = ttml::core::from_vector(test_data, {1, 1, 1, 4}, &device->get_device());

    auto vec_back = ttml::core::to_vector(tensor);

    ASSERT_EQ(vec_back.size(), test_data.size());
    EXPECT_EQ(vec_back, test_data);
}