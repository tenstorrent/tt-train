#include "trivial_ttnn_ops.hpp"

#include "core/tt_tensor_utils.hpp"
#include "core/ttnn_all_includes.hpp"

namespace ttml::ttnn_fixed {

tt::tt_metal::Tensor sum_over_batch(const tt::tt_metal::Tensor& t) {
    auto res = ttnn::moreh_sum(t, /* dim */ 0, /* keep_dim */ true, std::nullopt, std::nullopt, std::nullopt);
    return res;
}

tt::tt_metal::Tensor max(const tt::tt_metal::Tensor& t, int dim, bool keepdim) {
    const float kMinValue = -200.F;
    auto mask = core::ones(t.get_shape(), t.device());
    auto masked_t = ttnn::where(mask, t, kMinValue);
    auto res = ttnn::max(masked_t, dim, keepdim);
    return res;
}

tt::tt_metal::Tensor softmax(const tt::tt_metal::Tensor& t, int dim) {
    auto max_t = max(t, dim, /* keepdim */ true);
    auto t_sub_max = ttnn::subtract(t, max_t);
    return ttnn::softmax(t_sub_max, dim);
}

}  // namespace ttml::ttnn_fixed
