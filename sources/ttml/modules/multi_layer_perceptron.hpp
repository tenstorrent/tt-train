// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "autograd/module_base.hpp"
#include "modules/linear_module.hpp"
#include "ops/unary_ops.hpp"

namespace ttml::modules {

struct MultiLayerPerceptronParameters {
    uint32_t m_input_features{};
    std::vector<uint32_t> m_hidden_features;
    uint32_t m_output_features{};
};

class MultiLayerPerceptron : public autograd::ModuleBase {
private:
    std::vector<std::shared_ptr<LinearLayer>> m_layers;

public:
    explicit MultiLayerPerceptron(const MultiLayerPerceptronParameters& params);

    [[nodiscard]] autograd::TensorPtr operator()(autograd::TensorPtr tensor);
};

}  // namespace ttml::modules