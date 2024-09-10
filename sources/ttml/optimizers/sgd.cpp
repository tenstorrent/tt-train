#include "sgd.hpp"

namespace ttml::optimizers {

SGD::SGD(ttml::autograd::NamedParameters parameters, const SGDConfig& config) :
    m_lr(config.lr), m_momentum(config.momentum), m_parameters(std::move(parameters)) {
    for (const auto& [name, tensor_ptr] : m_parameters) {
        if (tensor_ptr->get_require_grad()) {
            m_theta.emplace(name, core::zeros_like(tensor_ptr->get_value()));
        }
    }
}

void SGD::zero_grad() {
    for (auto& [name, tensor_ptr] : m_parameters) {
        if (tensor_ptr->get_require_grad() && tensor_ptr->is_grad_initialized()) {
            auto& grad = tensor_ptr->get_grad();
            core::fill(grad, 0.0F);
        }
    }
}

void SGD::step() {
    for (auto& [name, theta] : m_theta) {
        auto tensor_ptr = m_parameters.at(name);

        theta = ttnn::multiply(theta, m_momentum);
        theta = ttnn::add(theta, ttnn::multiply(tensor_ptr->get_grad(), 1 - m_momentum));

        auto update = ttnn::multiply(theta, m_lr);
        tensor_ptr->set_value(ttnn::add(tensor_ptr->get_value(), update));
    }
}

}  // namespace ttml::optimizers