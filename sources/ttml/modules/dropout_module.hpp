#pragma once

#include <memory>

#include "autograd/auto_context.hpp"
#include "autograd/graph.hpp"
#include "autograd/module_base.hpp"
#include "autograd/tensor.hpp"
#include "ops/dropout_op.hpp"

namespace ttml::modules {

class DropoutLayer : public autograd::ModuleBase {
    std::string m_name;
    bool probability

        void
        initialize_tensors(uint32_t in_features, uint32_t out_features);

public:
    [[nodiscard]] const std::string& get_name() const;

    DropoutLayer(float probability);

    [[nodiscard]] autograd::TensorPtr operator()(const autograd::TensorPtr& tensor);
};

}  // namespace ttml::modules