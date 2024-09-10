#include "sgd.hpp"

#include <fmt/format.h>

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
            fmt::print("Cleaning gradients for {}\n", name);
            auto& grad = tensor_ptr->get_grad();
            core::fill(grad, 0.0F);
        }
    }
}

void SGD::step() {
    for (auto& [name, theta] : m_theta) {
        auto tensor_ptr = m_parameters.at(name);

        theta = ttnn::multiply(theta, m_momentum);
        theta = ttnn::subtract(theta, ttnn::multiply(tensor_ptr->get_grad(), 1 - m_momentum));

        auto update = ttnn::multiply(theta, m_lr);

        fmt::print("Updating {}\n", name);
        auto param_vec = core::to_vector(tensor_ptr->get_value());
        auto update_vec = core::to_vector(update);

        tensor_ptr->set_value(ttnn::add(tensor_ptr->get_value(), update));
        auto param_after_update_vec = core::to_vector(tensor_ptr->get_value());
        for (size_t i = 0; i < param_vec.size(); ++i) {
            fmt::print(
                "Param: {} Update: {} Param after update: {}\n",
                param_vec[i],
                update_vec[i],
                param_after_update_vec[i]);
        }
        fmt::print("**************\n");
    }
}

}  // namespace ttml::optimizers