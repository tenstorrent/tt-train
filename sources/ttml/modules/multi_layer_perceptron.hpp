#pragma once

#include <vector>

#include "autograd/module_base.hpp"
#include "modules/linear_module.hpp"
#include "ops/unary_ops.hpp"

namespace ttml::modules {

struct MultiLayerPerceptronParameters {
    uint32_t m_num_hidden_layers{};
    uint32_t m_input_features{};
    uint32_t m_hidden_features{};
    uint32_t m_output_features{};
};

class MultiLayerPerceptron : public autograd::ModuleBase {
private:
    std::vector<LinearLayer> m_layers;

public:
    explicit MultiLayerPerceptron(const MultiLayerPerceptronParameters& params);

    [[nodiscard]] autograd::TensorPtr operator()(autograd::TensorPtr tensor);
};

}  // namespace ttml::modules