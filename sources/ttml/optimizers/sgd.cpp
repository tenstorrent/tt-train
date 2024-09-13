#include "sgd.hpp"

#include <fmt/format.h>

#include <ttnn/operations/eltwise/binary/binary.hpp>

#include "core/tt_tensor_utils.hpp"

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
            grad = core::zeros_like(grad);
        }
    }
}

void SGD::step() {
    for (auto& [name, theta] : m_theta) {
        auto tensor_ptr = m_parameters.at(name);

        theta = ttnn::multiply(theta, m_momentum);
        theta = ttnn::add(theta, tensor_ptr->get_grad());
        auto tensor_update = ttnn::multiply(theta, m_lr);
        tensor_ptr->set_value(ttnn::subtract(tensor_ptr->get_value(), tensor_update));
    }
}

}  // namespace ttml::optimizers