#pragma once

#include <ttnn/tensor/tensor.hpp>

#include "autograd/module_base.hpp"
#include "core/tensor_utils.hpp"

namespace ttml::optimizers {

struct SGDParams {
    float lr;
    float momentum;
};

class SGD {
public:
    explicit SGD(ttml::autograd::NamedParameters parameters, const SGDParams& params) :
        m_lr(params.lr), m_momentum(params.momentum), m_parameters(std::move(parameters)) {
        for (const auto& [name, tensor_ptr] : m_parameters) {
            m_theta.emplace(name, core::zeros_like(tensor_ptr->get_grad()));
        }
    }

    void zero_grad() {
        for (auto& [name, tensor_ptr] : m_parameters) {
            auto& grad = tensor_ptr->get_grad();
            core::fill(grad, 0.0F);
        }
    }

    void step() {
        for (auto& [name, theta] : m_theta) {
            auto tensor_ptr = m_parameters.at(name);

            theta = ttnn::multiply(theta, m_momentum);
            theta = ttnn::add(theta, ttnn::multiply(tensor_ptr->get_grad(), 1 - m_momentum));

            auto update = ttnn::multiply(theta, m_lr);
            tensor_ptr->set_value(ttnn::add(tensor_ptr->get_value(), update));
        }
    }

private:
    float m_lr;
    float m_momentum;
    ttml::autograd::NamedParameters m_parameters;
    std::unordered_map<std::string, tt::tt_metal::Tensor> m_theta;
}

}  // namespace ttml::optimizers