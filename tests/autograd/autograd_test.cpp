

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/device.hpp"
#include "core/tensor_utils.hpp"
#include "core/ttnn_all_includes.hpp"
#include "ops/binary_ops.hpp"

class AutogradTest : public ::testing::Test {
protected:
    void SetUp() override { device = std::make_unique<ttml::core::Device>(0); }

    void TearDown() override {
        ttml::autograd::ctx().reset_graph();
        device.reset();
    }

    std::unique_ptr<ttml::core::Device> device;
};

TEST_F(AutogradTest, TestSum) {
    using namespace ttml::ops;
    std::vector<float> test_data1 = {1.F, 2.F, 3.F, 4.F};
    std::vector<float> test_data2 = {4.F, 3.F, 2.F, 1.F};
    auto tensor1 = ttml::core::from_vector(test_data1, {1, 1, 1, 4}, &device->get_device());
    auto tensor2 = ttml::core::from_vector(test_data1, {1, 1, 1, 4}, &device->get_device());

    auto t1 = std::make_shared<ttml::autograd::Tensor>(tensor1);
    auto t2 = std::make_shared<ttml::autograd::Tensor>(tensor2);

    auto res = t1 + t2;
    res->backward();
    auto res_back = ttml::core::to_vector(res->get_grad());
    auto t1_back = ttml::core::to_vector(t1->get_grad());
    auto t2_back = ttml::core::to_vector(t2->get_grad());

    for (float it : res_back) {
        EXPECT_EQ(it, 1.0F);
    }
    for (float it : t1_back) {
        EXPECT_EQ(it, 1.0F);
    }
    for (float it : t2_back) {
        EXPECT_EQ(it, 1.0F);
    }
}

TEST_F(AutogradTest, TestMul) {
    using namespace ttml::ops;
    std::vector<float> test_data1 = {1.F, 2.F, 3.F, 4.F};
    std::vector<float> test_data2 = {4.F, 3.F, 2.F, 1.F};
    auto tensor1 = ttml::core::from_vector(test_data1, {1, 1, 1, 4}, &device->get_device());
    auto tensor2 = ttml::core::from_vector(test_data2, {1, 1, 1, 4}, &device->get_device());

    auto t1 = std::make_shared<ttml::autograd::Tensor>(tensor1);
    auto t2 = std::make_shared<ttml::autograd::Tensor>(tensor2);

    auto res = t1 * t2;
    res->backward();
    auto res_back = ttml::core::to_vector(res->get_grad());
    auto t1_back = ttml::core::to_vector(t1->get_grad());
    auto t2_back = ttml::core::to_vector(t2->get_grad());

    for (float it : res_back) {
        EXPECT_EQ(it, 1.0F);
    }
    EXPECT_EQ(t2_back, test_data1);
    EXPECT_EQ(t1_back, test_data2);
}