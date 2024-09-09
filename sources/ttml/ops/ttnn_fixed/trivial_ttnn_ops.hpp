#pragma once
#include <ttnn/tensor/tensor.hpp>

#include "core/ttnn_all_includes.hpp"

namespace ttml::ops::ttnn_fixed {
tt::tt_metal::Tensor sum_over_batch(const tt::tt_metal::Tensor& t);

std::array<tt::tt_metal::Tensor, 2> add_bw(
    const tt::tt_metal::Tensor& grad, const tt::tt_metal::Tensor& a, const tt::tt_metal::Tensor& b);
}  // namespace ttml::ops::ttnn_fixed
