#include "autocast_tensor.hpp"

#include "core/tt_tensor_utils.hpp"

namespace ttml::autograd {

ttnn::Shape AutocastTensor::get_shape() const {
    return m_full_precision_tensor.get_shape();
}

void AutocastTensor::set_tensor(const tt::tt_metal::Tensor &tensor) {
    if (tensor.get_dtype() == DataType::FLOAT32) {
        m_is_float32 = true;
        m_full_precision_tensor = tensor;
        m_half_precision_tensor = ttnn::typecast(tensor, DataType::BFLOAT16);
        return;
    }

    m_is_float32 = false;
    m_full_precision_tensor = tensor;
    m_half_precision_tensor = ttnn::Tensor();  // Reset the half precision tensor
}

const tt::tt_metal::Tensor &AutocastTensor::get_tensor(bool half_precision) const {
    if (half_precision && m_is_float32) {
        return m_half_precision_tensor;
    }

    return m_full_precision_tensor;
}

tt::tt_metal::Tensor &AutocastTensor::get_mutable_tensor(bool half_precision) {
    if (half_precision && m_is_float32) {
        throw std::runtime_error("AutocastTensor doesn't return non-const reference to half precision tensor");
    }

    return m_full_precision_tensor;
}

AutocastTensor::AutocastTensor(const tt::tt_metal::Tensor &tensor) {
    set_tensor(tensor);
}

}  // namespace ttml::autograd
