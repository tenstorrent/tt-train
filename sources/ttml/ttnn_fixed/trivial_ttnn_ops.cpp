#include "trivial_ttnn_ops.hpp"

#include <ttnn/operations/moreh/moreh_sum/moreh_sum.hpp>

#include "core/tt_tensor_utils.hpp"
#include "core/ttnn_all_includes.hpp"

namespace ttml::ttnn_fixed {

tt::tt_metal::Tensor sum_over_batch(const tt::tt_metal::Tensor& t) {
    auto res = ttnn::moreh_sum(t, /* dim */ 0, /* keep_dim */ true, std::nullopt, std::nullopt, std::nullopt);
    return res;
}

// This is a workaround for the lack of working `ttnn::max` implementation.
tt::tt_metal::Tensor max(const tt::tt_metal::Tensor& t, int dim, bool keepdim) {
    const float kMinValue = -10000.F;
    auto mask = core::ones(t.get_shape(), t.device());
    auto masked_t = ttnn::where(ttnn::eq(mask, 1.F), t, kMinValue);
    auto res = ttnn::max(masked_t, dim, keepdim);
    return res;
}

// Stable softmax implementation
// ttnn::softmax also exists, but it is not stable (even after max subtraction optimization)
tt::tt_metal::Tensor softmax(const tt::tt_metal::Tensor& t, int dim) {
    auto t_max = max(t, dim, /* keepdim */ true);
    auto t_sub_max = ttnn::subtract(t, t_max);
    auto t_sub_max_exp = ttnn::exp(t_sub_max);
    auto t_sum_over_dim =
        ttnn::moreh_sum(t_sub_max_exp, dim, /* keep_dim */ true, std::nullopt, std::nullopt, std::nullopt);
    auto inv_t_sum_over_dim = ttnn::reciprocal(/* queue_id */ 0, t_sum_over_dim);
    return ttnn::multiply(t_sub_max_exp, inv_t_sum_over_dim);
}

}  // namespace ttml::ttnn_fixed
