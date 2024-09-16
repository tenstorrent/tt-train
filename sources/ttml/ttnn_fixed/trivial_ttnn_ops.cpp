#include "trivial_ttnn_ops.hpp"

#include "core/ttnn_all_includes.hpp"

namespace ttml::ttnn_fixed {

tt::tt_metal::Tensor sum_over_batch(const tt::tt_metal::Tensor& t) {
    auto res = ttnn::moreh_sum(t, /* dim */ 0, /* keep_dim */ true, std::nullopt, std::nullopt, std::nullopt);
    return res;
}

}  // namespace ttml::ttnn_fixed
