#include "multi_layer_perceptron.hpp"

namespace ttml::modules {

MultiLayerPerceptron::MultiLayerPerceptron(const MultiLayerPerceptronParameters& params) : autograd::ModuleBase() {
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
        register_module(layer.shared_from_this(), layer.get_name());
    }
}
autograd::TensorPtr MultiLayerPerceptron::operator()(autograd::TensorPtr tensor) {
    for (size_t index = 0; index < m_layers.size(); ++index) {
        tensor = m_layers[index](tensor);
        if (index + 1 != m_layers.size()) {
            tensor = ttml::ops::relu(tensor);
        }
    }

    return tensor;
}

}  // namespace ttml::modules