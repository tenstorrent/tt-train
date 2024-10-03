#include "adamw.hpp"

#include "autograd/module_base.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

namespace ttml::optimizers {

AdamW::AdamW(autograd::NamedParameters parameters, const AdamWConfig& config) :
    m_config(config), m_parameters(std::move(parameters)) {
    for (const auto& [key, tensor_ptr] : m_parameters) {
        if (tensor_ptr->get_require_grad()) {
            m_first_moment.emplace(key, core::zeros_like(tensor_ptr->get_value()));
            m_second_moment.emplace(key, core::zeros_like(tensor_ptr->get_value()));
        }
    }
}

void AdamW::zero_grad() {
    for (auto& [key, tensor_ptr] : m_parameters) {
        if (tensor_ptr->get_require_grad() && tensor_ptr->is_grad_initialized()) {
            tensor_ptr->set_grad(core::zeros_like(tensor_ptr->get_value()));
        }
    }
}

void AdamW::step() {
    steps++;
    for (auto& [key, first_moment] : m_first_moment) {
        auto tensor_ptr = m_parameters.at(key);

        auto gradients = tensor_ptr->get_grad();
        if (m_config.weight_decay != 0.0F) {
            auto weight_decay_update = ttnn::multiply(tensor_ptr->get_value(), m_config.weight_decay * m_config.lr);
            tensor_ptr->set_value(ttnn::subtract(tensor_ptr->get_value(), weight_decay_update));
        }

        first_moment =
            ttnn::add(ttnn::multiply(first_moment, m_config.beta1), ttnn::multiply(gradients, 1.F - m_config.beta1));
        auto& second_moment = m_second_moment.at(key);
        second_moment = ttnn::add(
            ttnn::multiply(second_moment, m_config.beta2),
            ttnn::multiply(ttnn::square(gradients), 1.F - m_config.beta2));
        auto first_moment_hat = ttnn::multiply(first_moment, 1.F / (1.F - std::pow(m_config.beta1, steps)));
        auto second_moment_hat = ttnn::multiply(second_moment, 1.F / (1.F - std::pow(m_config.beta2, steps)));
        tensor_ptr->set_value(ttnn::subtract(
            tensor_ptr->get_value(),
            ttnn_fixed::divide(
                ttnn::multiply(first_moment_hat, m_config.lr),
                ttnn::add(ttnn::sqrt(second_moment_hat), m_config.epsilon))));
    }
}

}  // namespace ttml::optimizers