#pragma once

#include <vector>

#include "autograd/module_base.hpp"
#include "modules/linear_module.hpp"
#include "ops/unary_ops.hpp"

namespace ttml::modules {

struct MultiLayerPerceptronParameters {
    uint32_t m_num_hidden_layers;
    uint32_t m_input_features;
    uint32_t m_hidden_features;
    uint32_t m_output_features;
};

class MultiLayerPerceptron : public autograd::ModuleBase {
private:
    std::vector<LinearLayer> m_layers;

public:
    explicit MultiLayerPerceptron(const MultiLayerPerceptronParameters& params) : ttml::autograd::ModuleBase() {
        if (params.m_num_hidden_layers == 0U) {
            m_layers.emplace_back(params.m_input_features, params.m_output_features);
        } else {
            m_layers.reserve(2U + params.m_num_hidden_layers);
            m_layers.emplace_back(params.m_input_features, params.m_hidden_features);
            for (uint32_t index = 0; index < params.m_num_hidden_layers; ++index) {
                m_layers.emplace_back(params.m_hidden_features, params.m_hidden_features);
            }
            m_layers.emplace_back(params.m_hidden_features, params.m_output_features);
        }
        create_name("mlp");
        for (auto& layer : m_layers) {
            register_module(layer.get_name(), layer.shared_from_this());
        }
    }

    autograd::TensorPtr operator()(autograd::TensorPtr tensor) {
        for (size_t index = 0; index < m_layers.size(); ++index) {
            tensor = m_layers[index](tensor);
            if (index + 1 != m_layers.size()) {
                tensor = ttml::ops::relu(tensor);
            }
        }

        return tensor;
    }
}

}  // namespace ttml::modules