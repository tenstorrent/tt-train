#include "layernorm_op.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <ttnn/deprecated/tt_dnn/op_library/moreh_layernorm/moreh_layernorm_op.hpp>
#include <ttnn/deprecated/tt_dnn/op_library/moreh_layernorm_backward/moreh_layernorm_backward_op.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/eltwise/unary/unary.hpp>
#include <ttnn/tensor/tensor.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/graph.hpp"
#include "autograd/graph_utils.hpp"
#include "core/tt_tensor_utils.hpp"
#include "core/ttnn_all_includes.hpp"

namespace ttml::ops {

// simplified version of layernorm
// it works only for 4D tensors and for the last dimension
autograd::TensorPtr layernorm(
    const autograd::TensorPtr& tensor, const autograd::TensorPtr& gamma, const autograd::TensorPtr& beta) {
    auto tensor_shape = tensor->get_value().get_shape();
    auto mean = core::zeros(
        core::create_shape({tensor_shape[0], tensor_shape[1], tensor_shape[2], 1}), &autograd::ctx().get_device());
    auto rstd = core::zeros(
        core::create_shape({tensor_shape[0], tensor_shape[1], tensor_shape[2], 1}), &autograd::ctx().get_device());
    auto output = core::zeros_like(tensor->get_value());

    auto out_tensors = tt::operations::primary::moreh_layernorm(
        tensor->get_value(),
        1,
        1e-6F,
        /* gamma */ gamma->get_value(),
        /* beta */ beta->get_value(),
        output,
        mean,
        rstd);

    auto out = autograd::create_tensor();
    out->set_value(out_tensors[0].value());
    mean = out_tensors[1].value();
    rstd = out_tensors[2].value();

    autograd::GradFunction grad = [tensor, out, mean, rstd, gamma, beta]() {
        auto input_grad = core::zeros_like(tensor->get_value());
        auto gamma_grad = core::zeros_like(gamma->get_value());
        auto beta_grad = core::zeros_like(beta->get_value());

        auto res = tt::operations::primary::moreh_layernorm_backward(
            out->get_grad(), tensor->get_value(), mean, rstd, 1, gamma->get_value(), input_grad, gamma_grad, beta_grad);

        tensor->add_grad(res[0].value());
        gamma->add_grad(res[1].value());
        beta->add_grad(res[2].value());
    };

    auto links = autograd::get_links(tensor);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}
}  // namespace ttml::ops