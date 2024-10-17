// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "adamw.hpp"

#include "autograd/module_base.hpp"
#include "core/tt_tensor_utils.hpp"
#include "optimizers/optimizer_base.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

namespace {

const std::string kFirstMoment = "first_moment/";
const std::string kSecondMoment = "second_moment/";

}  // namespace

namespace ttml::optimizers {

AdamW::AdamW(autograd::NamedParameters parameters, const AdamWConfig& config) :
    m_config(config), m_parameters(std::move(parameters)) {
    for (const auto& [key, tensor_ptr] : m_parameters) {
        if (tensor_ptr->get_requires_grad()) {
            m_first_moment.emplace(
                key, autograd::create_tensor(core::zeros_like(tensor_ptr->get_value()), /* requires_grad */ false));
            m_second_moment.emplace(
                key, autograd::create_tensor(core::zeros_like(tensor_ptr->get_value()), /* requires_grad */ false));
        }
    }
}

void AdamW::zero_grad() {
    for (auto& [key, tensor_ptr] : m_parameters) {
        if (tensor_ptr->get_requires_grad() && tensor_ptr->is_grad_initialized()) {
            tensor_ptr->set_grad(core::zeros_like(tensor_ptr->get_value()));
        }
    }
}

void AdamW::step() {
    m_steps++;
    for (auto& [key, first_moment_ptr] : m_first_moment) {
        const auto& tensor_ptr = m_parameters.at(key);
        if (!tensor_ptr->is_grad_initialized()) {
            continue;
        }
        auto& second_moment_ptr = m_second_moment.at(key);
        auto& first_moment = first_moment_ptr->get_value();
        auto& second_moment = second_moment_ptr->get_value();

        const auto& gradients = tensor_ptr->get_grad();
        ttnn::moreh_adamw(
            tensor_ptr->get_value(),
            gradients,
            first_moment,
            second_moment,
            m_config.lr,
            m_config.beta1,
            m_config.beta2,
            m_config.epsilon,
            m_config.weight_decay,
            m_steps,
            /* amsgrad */ false,
            /* max_exp_avg_sq_in */ std::nullopt,
            /* param_out */ tensor_ptr->get_value(),
            /* exp_avg_out */ first_moment,
            /* exp_avg_sq_out */ second_moment,
            /* max_exp_avg_sq_out */ std::nullopt,
            /* memory_config */ std::nullopt,
            /* compute_kernel_config */ std::nullopt);
    }
}

[[nodiscard]] autograd::NamedParameters AdamW::get_state_dict() const {
    autograd::NamedParameters state_dict;
    for (const auto& [key, first_moment] : m_first_moment) {
        state_dict.emplace(kFirstMoment + key, first_moment);
    }

    for (const auto& [key, second_moment] : m_second_moment) {
        state_dict.emplace(kSecondMoment + key, second_moment);
    }

    return state_dict;
}

void AdamW::set_state_dict(const autograd::NamedParameters& dict) {
    for (const auto& [key, tensor] : dict) {
        if (key.starts_with(kFirstMoment)) {
            m_first_moment[key.substr(kFirstMoment.size())] = tensor;
        } else if (key.starts_with(kSecondMoment)) {
            m_second_moment[key.substr(kSecondMoment.size())] = tensor;
        } else {
            throw std::runtime_error(fmt::format("AdamW: Invalid key in state dict. Key = {}", key));
        }
    }
}

[[nodiscard]] size_t AdamW::get_steps() const {
    return m_steps;
}

void AdamW::set_steps(size_t steps) {
    m_steps = steps;
}

}  // namespace ttml::optimizers
