// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ttnn/tensor/tensor.hpp>

#include "autograd/module_base.hpp"
#include "core/tt_tensor_utils.hpp"

namespace ttml::optimizers {

class OptimizerBase {
public:
    OptimizerBase() = default;
    OptimizerBase(const OptimizerBase&) = delete;
    OptimizerBase& operator=(const OptimizerBase&) = delete;
    OptimizerBase(OptimizerBase&&) = delete;
    OptimizerBase& operator=(OptimizerBase&&) = delete;
    virtual ~OptimizerBase() = default;

    virtual void zero_grad() = 0;

    virtual void step() = 0;

    [[nodiscard]] virtual autograd::NamedParameters get_state_dict() const = 0;
    virtual void set_state_dict(const autograd::NamedParameters& dict) = 0;

    [[nodiscard]] virtual size_t get_steps() const = 0;
    virtual void set_steps(size_t steps) = 0;
};

}  // namespace ttml::optimizers