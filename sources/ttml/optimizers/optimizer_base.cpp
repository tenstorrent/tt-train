// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "optimizer_base.hpp"

#include "core/tt_tensor_utils.hpp"

namespace ttml::optimizers {

void OptimizerBase::print_stats() const {
    fmt::print("\n\nOptimization parameters values and gradients:\n");
    for (const auto& [name, tensor] : m_parameters) {
        core::print_tensor_stats(tensor->get_value(), fmt::format("{}/value", name));
        if (tensor->is_grad_initialized()) {
            core::print_tensor_stats(tensor->get_grad(), fmt::format("{}/gradient", name));
        }
    }
    fmt::print("=================================================\n");
}

}  // namespace ttml::optimizers
