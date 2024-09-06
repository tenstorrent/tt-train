#pragma once

#include <memory>

#include "autograd/auto_context.hpp"
#include "autograd/graph.hpp"
#include "autograd/module_base.hpp"
#include "autograd/tensor.hpp"

namespace ttml::modules {

class LinearLayer : public autograd::ModuleBase {
    std::string m_name;
    autograd::TensorPtr m_weight;
    autograd::TensorPtr m_bias;
    autograd::GradFunction backward;

    // TODO: finish initialization
    void initialize_tensors([[maybe_unused]] uint32_t in_features, [[maybe_unused]] uint32_t out_features) {}

public:
    const std::string& get_name() const { return m_name; }

    LinearLayer(uint32_t in_features, uint32_t out_features) {
        initialize_tensors(in_features, out_features);

        create_name("linear");
        register_tensor("weight", m_weight);
        register_tensor("bias", m_bias);
    }

    autograd::TensorPtr operator()(const autograd::TensorPtr& tensor) {
        autograd::TensorPtr out;
        out->set_value(ttnn::linear(tensor->get_value(), m_weight->get_value(), m_bias->get_value()));

        autograd::GradFunction grad = [weight = m_weight, bias = m_bias, tensor, out]() {
            /// TODO: implement backward
        };

        std::vector<autograd::NodeId> links;
        if (m_weight->get_node().has_value()) {
            links.push_back(m_weight->get_node().value());
        }
        if (m_bias->get_node().has_value()) {
            links.push_back(m_bias->get_node().value());
        }
        if (tensor->get_node().has_value()) {
            links.push_back(tensor->get_node().value());
        }

        out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));
    }
};

}  // namespace ttml::modules