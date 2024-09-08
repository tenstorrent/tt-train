

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <memory>
#include <ttnn/tensor/types.hpp>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/device.hpp"
#include "core/tt_tensor_utils.hpp"
#include "core/ttnn_all_includes.hpp"
#include "ops/binary_ops.hpp"
#include "ops/unary_ops.hpp"
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

TEST_F(AutogradTest, BroadCastBatchTest) {
    using namespace ttml::ops;
    std::vector<float> test_data1 = {1.F, 2.F, 3.F, 4.F};
    tt::tt_metal::Shape shape = {1, 1, 1, 4};
    auto tensor1 = ttml::core::from_vector(test_data1, shape, &device->get_device());
    auto t1 = std::make_shared<ttml::autograd::Tensor>(tensor1);
    uint32_t new_batch = 4;
    auto res = ttml::ops::broadcast_batch(t1, new_batch);
    res->backward();
    auto t1_back = ttml::core::to_vector(t1->get_grad());
    ttnn::Shape batch_shape(std::array<uint32_t, 4>{4, 1, 1, 4});
    ttnn::Shape new_shape = res->get_value().get_shape();
    ttnn::Shape back_shape = t1->get_grad().get_shape();

    for (size_t i = 0; i < 4; i++) {
        EXPECT_EQ(new_shape[i], batch_shape[i]);
    }
    for (size_t i = 0; i < 4; i++) {
        EXPECT_EQ(back_shape[i], shape[i]);
    }
    for (size_t i = 0; i < 4; i++) {
        EXPECT_EQ(t1_back[i], test_data1[i] * new_batch);
    }
}