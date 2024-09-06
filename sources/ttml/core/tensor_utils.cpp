#include "core/tensor_utils.hpp"

#include <ttnn/operations/creation.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/eltwise/unary_backward/unary_backward.hpp>

namespace ttml::core {

tt::tt_metal::Tensor zeros_like(const tt::tt_metal::Tensor& tensor) { return ttnn::zeros_like(tensor); }

void fill(tt::tt_metal::Tensor& tensor, const float value) {
    // TODO: optimize to do it in one operation
    tensor = ttnn::multiply(ttnn::ones_like(tensor), value);
}

}  // namespace ttml::core