#include "trivial_ttnn_ops.hpp"

#include <array>

namespace ttml::ops::ttnn_fixed {
tt::tt_metal::Tensor sum_over_batch(const tt::tt_metal::Tensor& t) {
    auto res = ttnn::sum(t, 0);
    auto output_shape = res.get_legacy_shape();
    res = res.reshape(output_shape);
    return res;
}
std::array<tt::tt_metal::Tensor, 2> add_bw(
    const tt::tt_metal::Tensor& grad, const tt::tt_metal::Tensor& a, const tt::tt_metal::Tensor& b) {
    std::array<tt::tt_metal::Tensor, 2> res = {grad, grad};
    if (grad.get_legacy_shape()[0] > a.get_legacy_shape()[0]) {
        res[0] = sum_over_batch(grad);
    }
    if (grad.get_legacy_shape()[0] > b.get_legacy_shape()[0]) {
        res[1] = sum_over_batch(grad);
    }
    return res;
}
}  // namespace ttml::ops::ttnn_fixed
