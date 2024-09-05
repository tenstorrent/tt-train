#pragma once

#include <memory>

#include "autograd/graph.hpp"
#include "autograd/module_base.hpp"
#include "autograd/tensor.hpp"

namespace ttml::modules {

class LinearLayer : public autograd::ModuleBase {
    std::string m_name;
    autograd::TensorPtr m_weight;
    autograd::TensorPtr m_bias;
    autograd::GradFunction backward;

    void initialize_tensors([[maybe_unused]] uint32_t in_features, [[maybe_unused]] uint32_t out_features) {}

public:
    const std::string& get_name() const { return m_name; }

    LinearLayer(uint32_t in_features, uint32_t out_features) {
        initialize_tensors(in_features, out_features);

        create_name("linear");
        register_tensor("weight", m_weight);
        register_tensor("bias", m_bias);
    }

    // TODO: finish implementation of the layer
    autograd::TensorPtr operator()(const autograd::TensorPtr& tensor) { return tensor; }
};

}  // namespace ttml::modules