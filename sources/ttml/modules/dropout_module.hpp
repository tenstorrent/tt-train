// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

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
    float m_prob = 0.2F;

public:
    explicit DropoutLayer(float probability);

    [[nodiscard]] autograd::TensorPtr operator()(const autograd::TensorPtr& tensor);
};

}  // namespace ttml::modules