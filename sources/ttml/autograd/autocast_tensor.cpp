#include "autocast_tensor.hpp"

#include "core/tt_tensor_utils.hpp"

namespace ttml::autograd {

ttnn::Shape AutocastTensor::get_shape() const {
    return m_full_precision_tensor.get_shape();
}

void AutocastTensor::set_tensor(const tt::tt_metal::Tensor &tensor) {
    if (tensor.get_dtype() == DataType::FLOAT32) {
        m_is_float32 = true;
        m_needs_update = true;
        m_full_precision_tensor = tensor;
        m_half_precision_tensor = std::nullopt;
        return;
    }

    m_is_float32 = false;
    m_needs_update = false;
    m_full_precision_tensor = tensor;
    m_half_precision_tensor = std::nullopt;
}

tt::tt_metal::Tensor &AutocastTensor::get_half_precision_tensor() {
    if (!m_is_float32) {
        return m_full_precision_tensor;
    }

    if (m_needs_update) {
        m_half_precision_tensor = ttnn::typecast(m_full_precision_tensor, DataType::BFLOAT16);
        m_needs_update = false;
    }

    return m_half_precision_tensor.value();
}

tt::tt_metal::Tensor &AutocastTensor::get_tensor(bool half_precision) {
    if (half_precision) {
        return get_half_precision_tensor();
    }

    return m_full_precision_tensor;
}

AutocastTensor::AutocastTensor(const tt::tt_metal::Tensor &tensor) {
    set_tensor(tensor);
}

}  // namespace ttml::autograd
