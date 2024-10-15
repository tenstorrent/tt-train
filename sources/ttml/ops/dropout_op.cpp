// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dropout_op.hpp"

#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/eltwise/unary/unary.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/graph.hpp"
#include "autograd/graph_utils.hpp"
#include "core/tt_tensor_utils.hpp"
#include "core/ttnn_all_includes.hpp"

namespace ttml::ops {

autograd::TensorPtr dropout(const autograd::TensorPtr& tensor, float probability) {
    auto mask = core::ones_like(tensor->get_value());
    // dropout seed is not properly used in ttnn::dropout
    auto dropout_seed = autograd::ctx().get_generator()();
    auto scaler = 1.0F / (1.0F - probability);
    mask = ttnn::dropout(mask, dropout_seed, probability, scaler);
    auto out = autograd::create_tensor();
    auto masked_out = ttnn::multiply(tensor->get_value(), mask);
    out->set_value(masked_out);
    autograd::GradFunction grad = [tensor, out, mask]() {
        auto res = ttnn::multiply(out->get_grad(), mask);
        tensor->add_grad(res);
    };

    auto links = autograd::get_links(tensor);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}

}  // namespace ttml::ops