#pragma once

#include "core/ttnn_all_includes.hpp"

namespace ttml::core {

tt::tt_metal::Tensor zeros_like(const tt::tt_metal::Tensor& tensor);
void fill(tt::tt_metal::Tensor& tensor, const float value);

}  // namespace ttml::core