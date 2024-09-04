#include "tensor.hpp"

namespace ttml::autograd {
Tensor::Tensor(tt::tt_metal::Tensor m_value, tt::tt_metal::Tensor m_grad) :
    m_value(std::move(m_value)), m_grad(std::move(m_grad)) {}

void Tensor::add_grad(const tt::tt_metal::Tensor &grad) { m_grad = ttnn::add_(m_grad, grad); }
}  // namespace ttml::autograd