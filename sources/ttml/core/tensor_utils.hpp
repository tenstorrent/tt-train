#pragma once

#include <vector>

#include "core/ttnn_fwd.hpp"
namespace ttml::core {
// for now we are using bfloat16 type
void fill(tt::tt_metal::Tensor& tensor, const float value);

tt::tt_metal::Tensor zeros_like(const tt::tt_metal::Tensor& tensor);
tt::tt_metal::Tensor ones_like(const tt::tt_metal::Tensor& tensor);

tt::tt_metal::Tensor zeros(const tt::tt_metal::Shape& shape, tt::tt_metal::Device* device);
tt::tt_metal::Tensor ones(const tt::tt_metal::Shape& shape, tt::tt_metal::Device* device);
tt::tt_metal::Tensor from_vector(
    const std::vector<float>& buffer, const tt::tt_metal::Shape& shape, tt::tt_metal::Device* device);

std::vector<float> to_vector(const tt::tt_metal::Tensor& tensor);
}  // namespace ttml::core