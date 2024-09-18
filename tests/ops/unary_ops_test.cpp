#include "ops/unary_ops.hpp"

#include <gtest/gtest.h>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "core/ttnn_all_includes.hpp"

TEST(UnaryOpsTest, GlobalMean) {
    std::vector<float> test_data = {1.F, 2.F, 3.F, 4.F, 1.F, 2.F, 3.F, 4.F};

    auto shape = ttml::core::create_shape({2, 1, 1, 4});
    auto tensor = ttml::core::from_vector(test_data, shape, &ttml::autograd::ctx().get_device());

    auto tensor_ptr = std::make_shared<ttml::autograd::Tensor>(tensor);

    auto result = ttml::ops::mean(tensor_ptr);
    auto result_data = ttml::core::to_vector(result->get_value());

    ASSERT_EQ(result_data.size(), 1);
    EXPECT_FLOAT_EQ(result_data[0], 2.5F);

    result->backward();
    auto tensor_grad = ttml::core::to_vector(tensor_ptr->get_grad());
    ASSERT_EQ(tensor_grad.size(), test_data.size());
    for (float it : tensor_grad) {
        EXPECT_FLOAT_EQ(it, 0.125F);
    }
}
