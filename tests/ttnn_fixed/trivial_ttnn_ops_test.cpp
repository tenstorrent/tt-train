#include "ttnn_fixed/trivial_ttnn_ops.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/device.hpp"
#include "core/tt_tensor_utils.hpp"
#include "core/ttnn_all_includes.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

TEST(TrivialTnnFixedTest, TestMax_0) {
    auto* device = &ttml::autograd::ctx().get_device();

    std::vector<float> data(24, -1.F);
    tt::tt_metal::Shape shape = {1, 2, 3, 4};
    auto tensor = ttml::core::from_vector(data, ttnn::Shape(shape), device);
    auto res = ttml::ttnn_fixed::max(tensor, /* dim */ 3, /* keepdim */ true);
    auto res_vector = ttml::core::to_vector(res);
    EXPECT_EQ(res_vector.size(), 6);
    for (const auto& value : res_vector) {
        EXPECT_NEAR(value, -1.F, 1e-2);
    }
}

TEST(TrivialTnnFixedTest, TestMax_1) {
    auto* device = &ttml::autograd::ctx().get_device();

    tt::tt_metal::Shape shape = {4, 1, 1, 4};
    std::vector<float> data(16);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            data[i * 4 + j] = -static_cast<float>(i + 1);
        }
    }
    auto tensor = ttml::core::from_vector(data, ttnn::Shape(shape), device);
    auto res = ttml::ttnn_fixed::max(tensor, /* dim */ 3, /* keepdim */ true);
    auto res_vector = ttml::core::to_vector(res);
    EXPECT_EQ(res_vector.size(), 4);
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(res_vector[i], -static_cast<float>(i + 1), 1e-2);
    }
}

TEST(TrivialTnnFixedTest, TestStableSoftmax_0) {
    auto* device = &ttml::autograd::ctx().get_device();

    const size_t batch_size = 1U;
    const size_t features = 2U;
    std::vector<float> data(batch_size * features);
    for (int i = 0; i < data.size(); ++i) {
        data[i] = 100.F + static_cast<float>(i);
    }
    tt::tt_metal::Shape shape = {batch_size, 1, 1, features};
    auto tensor = ttml::core::from_vector(data, ttnn::Shape(shape), device);
    auto tensor_data = ttml::core::to_vector(tensor);
    EXPECT_NEAR(tensor_data[0], 100.F, 1e-2);
    EXPECT_NEAR(tensor_data[1], 101.F, 1e-2);

    auto res = ttml::ttnn_fixed::softmax(tensor, /* dim */ 3);
    auto res_vector = ttml::core::to_vector(res);
    EXPECT_NEAR(res_vector[0], 0.2689F, 2e-2);
    EXPECT_NEAR(res_vector[1], 0.7311F, 2e-2);
}

TEST(TrivialTnnFixedTest, TestStableSoftmax_1) {
    auto* device = &ttml::autograd::ctx().get_device();

    const size_t batch_size = 1U;
    const size_t features = 2U;
    std::vector<float> data(batch_size * features);
    for (int i = 0; i < data.size(); ++i) {
        data[i] = -100.F + static_cast<float>(i);
    }
    tt::tt_metal::Shape shape = {batch_size, 1, 1, features};
    auto tensor = ttml::core::from_vector(data, ttnn::Shape(shape), device);
    auto tensor_data = ttml::core::to_vector(tensor);
    EXPECT_NEAR(tensor_data[0], -100.F, 1e-2);
    EXPECT_NEAR(tensor_data[1], -99.F, 1e-2);

    auto res = ttml::ttnn_fixed::softmax(tensor, /* dim */ 3);
    auto res_vector = ttml::core::to_vector(res);
    EXPECT_NEAR(res_vector[0], 0.2689F, 2e-2);
    EXPECT_NEAR(res_vector[1], 0.7311F, 2e-2);
}

TEST(TrivialTnnFixedTest, TestSumOverBatch_0) {
    auto* device = &ttml::autograd::ctx().get_device();

    const size_t batch_size = 10U;
    const size_t features = 4U;
    std::vector<float> data(batch_size * features);
    std::iota(data.begin(), data.end(), 0);

    tt::tt_metal::Shape shape = {batch_size, 1, 1, features};
    auto tensor = ttml::core::from_vector(data, ttnn::Shape(shape), device);
    auto tensor_shape = tensor.get_shape();
    EXPECT_EQ(tensor_shape[0], batch_size);
    EXPECT_EQ(tensor_shape[1], 1U);
    EXPECT_EQ(tensor_shape[2], 1U);
    EXPECT_EQ(tensor_shape[3], features);

    auto result = ttml::ttnn_fixed::sum_over_batch(tensor);
    const auto& result_shape = result.get_shape();
    ASSERT_EQ(result_shape.rank(), 4U);
    EXPECT_EQ(result_shape[0], 1U);
    EXPECT_EQ(result_shape[1], 1U);
    EXPECT_EQ(result_shape[2], 1U);
    EXPECT_EQ(result_shape[3], features);
}

TEST(TrivialTnnFixedTest, TestSumOverBatch_1) {
    auto* device = &ttml::autograd::ctx().get_device();

    const size_t batch_size = 2U;
    const size_t features = 64U;
    std::vector<float> data(batch_size * features);
    float step = 0.1F;
    float value = 0.0F;
    for (int i = 0; i < data.size(); ++i) {
        data[i] = value;
        value += step;
    }

    tt::tt_metal::Shape shape = {batch_size, 1, 1, features};
    auto tensor = ttml::core::from_vector(data, ttnn::Shape(shape), device);
    auto tensor_shape = tensor.get_shape();
    EXPECT_EQ(tensor_shape[0], batch_size);
    EXPECT_EQ(tensor_shape[1], 1U);
    EXPECT_EQ(tensor_shape[2], 1U);
    EXPECT_EQ(tensor_shape[3], features);

    auto result = ttml::ttnn_fixed::sum_over_batch(tensor);
    const auto& result_shape = result.get_shape();
    ASSERT_EQ(result_shape.rank(), 4U);
    EXPECT_EQ(result_shape[0], 1U);
    EXPECT_EQ(result_shape[1], 1U);
    EXPECT_EQ(result_shape[2], 1U);
    EXPECT_EQ(result_shape[3], features);

    std::vector<float> resulting_vector = ttml::core::to_vector(result);
    EXPECT_EQ(resulting_vector.size(), features);
    const float eps = 1.0F;
    for (int i = 0; i < resulting_vector.size(); ++i) {
        float expected_value = 0.F;
        for (int j = 0; j < batch_size; ++j) {
            expected_value += static_cast<float>(i + j * features) * step;
        }

        EXPECT_NEAR(expected_value, resulting_vector[i], eps);
    }
}