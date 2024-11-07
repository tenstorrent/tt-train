// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "sgd.hpp"

#include <fmt/format.h>

#include "core/debug.hpp"
#include "core/tt_tensor_utils.hpp"

namespace ttml::optimizers {

SGD::SGD(ttml::autograd::NamedParameters parameters, const SGDConfig& config) :
    OptimizerBase(std::move(parameters)), m_config(config) {
    for (const auto& [name, tensor_ptr] : m_parameters) {
        if (tensor_ptr->get_requires_grad()) {
            m_theta.emplace(
                name,
                autograd::create_tensor(
                    core::zeros_like(tensor_ptr->get_value(autograd::Precision::FULL)), /* requires_grad */ false));
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
    if (core::debug::Debug::enable_print_tensor_stats()) {
        print_stats();
    }

    for (auto& [name, theta_ptr] : m_theta) {
        auto& theta = theta_ptr->get_mutable_value();
        const auto& tensor_ptr = m_parameters.at(name);
        if (!tensor_ptr->is_grad_initialized()) {
            continue;
        }

        auto gradients = tensor_ptr->get_grad();
        if (m_config.weight_decay != 0.0F) {
            gradients = ttnn::add(
                ttnn::multiply(tensor_ptr->get_value(autograd::Precision::FULL), m_config.weight_decay), gradients);
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
        tensor_ptr->set_value(
            ttnn::subtract(tensor_ptr->get_value(autograd::Precision::FULL), ttnn::multiply(gradients, m_config.lr)));
    }
    steps++;
}

autograd::NamedParameters SGD::get_state_dict() const {
    return m_theta;
}

void SGD::set_state_dict(const autograd::NamedParameters& dict) {
    m_theta = dict;
}

size_t SGD::get_steps() const {
    return steps;
}

void SGD::set_steps(size_t steps) {
    this->steps = steps;
}

}  // namespace ttml::optimizers
