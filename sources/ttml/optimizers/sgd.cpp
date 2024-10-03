#include "sgd.hpp"

#include <fmt/format.h>

#include <ttnn/operations/eltwise/binary/binary.hpp>

#include "core/tt_tensor_utils.hpp"

namespace ttml::optimizers {

SGD::SGD(ttml::autograd::NamedParameters parameters, const SGDConfig& config) :
    m_config(config), m_parameters(std::move(parameters)) {
    for (const auto& [name, tensor_ptr] : m_parameters) {
        if (tensor_ptr->get_requires_grad()) {
            m_theta.emplace(name, core::zeros_like(tensor_ptr->get_value()));
        }
    }
}

void SGD::zero_grad() {
    for (auto& [name, tensor_ptr] : m_parameters) {
        if (tensor_ptr->get_requires_grad() && tensor_ptr->is_grad_initialized()) {
            tensor_ptr->set_grad(core::zeros_like(tensor_ptr->get_value()));
        }
    }
}

void SGD::step() {
    for (auto& [name, theta] : m_theta) {
        auto tensor_ptr = m_parameters.at(name);

        auto gradients = tensor_ptr->get_grad();
        if (m_config.weight_decay != 0.0F) {
            gradients = ttnn::add(gradients, ttnn::multiply(tensor_ptr->get_value(), m_config.weight_decay));
        }

        if (m_config.momentum != 0.0F) {
            if (steps != 0) {
                // apply momentum
                theta = ttnn::multiply(theta, m_config.momentum);
                // dampening
                if (m_config.dampening != 0.0F) {
                    theta = ttnn::add(theta, ttnn::multiply(gradients, 1 - m_config.dampening));
                } else {
                    theta = ttnn::add(theta, gradients);
                }
            } else {
                theta = ttnn::add(theta, gradients);
            }

            if (m_config.nesterov) {
                gradients = ttnn::add(gradients, ttnn::multiply(theta, m_config.momentum));
            } else {
                gradients = theta;
            }
        }
        tensor_ptr->set_value(ttnn::subtract(tensor_ptr->get_value(), ttnn::multiply(gradients, m_config.lr)));
    }
    steps++;
}

TTTensorDict SGD::get_state_dict() const {
    return m_theta;
}

void SGD::set_state_dict(TTTensorDict dict) {
    m_theta = std::move(dict);
}

size_t SGD::get_steps() const {
    return steps;
}

void SGD::set_steps(size_t steps) {
    this->steps = steps;
}

}  // namespace ttml::optimizers