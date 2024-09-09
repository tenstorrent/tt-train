#pragma once
#include <ttnn/tensor/tensor.hpp>

#include "core/ttnn_all_includes.hpp"

namespace ttml::ttnn_fixed {

tt::tt_metal::Tensor sum_over_batch(const tt::tt_metal::Tensor& t);

}  // namespace ttml::ttnn_fixed
