#pragma once

#include <memory>

#include "autograd/auto_context.hpp"
#include "autograd/graph.hpp"
#include "autograd/module_base.hpp"
#include "autograd/tensor.hpp"
#include "ops/linear_op.hpp"

namespace ttml::modules {

class LinearLayer : public autograd::ModuleBase {
    std::string m_name;
    autograd::TensorPtr m_weight;
    autograd::TensorPtr m_bias;
    autograd::GradFunction backward;

    void initialize_tensors(uint32_t in_features, uint32_t out_features);

public:
    [[nodiscard]] const std::string& get_name() const;

    LinearLayer(uint32_t in_features, uint32_t out_features);

    [[nodiscard]] autograd::TensorPtr operator()(const autograd::TensorPtr& tensor);
};

}  // namespace ttml::modules