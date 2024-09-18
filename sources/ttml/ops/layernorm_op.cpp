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

namespace ttml::ttml::ops {

// simplified version of layernorm
// it works only for 4D tensors and for the last dimension
autograd::TensorPtr layernorm(
    const autograd::TensorPtr& tensor, const autograd::TensorPtr& gamma, const autograd::TensorPtr& beta) {
    autograd::TensorPtr out = autograd::create_tensor();
    auto tensor_shape = tensor->get_value().get_shape();
    tt::tt_metal::Tensor mean = core::zeros(
        core::create_shape({tensor_shape[0], tensor_shape[1], tensor_shape[2], 1}), &autograd::ctx().get_device());
    tt::tt_metal::Tensor rstd = core::zeros(
        core::create_shape({tensor_shape[0], tensor_shape[1], tensor_shape[2], 1}), &autograd::ctx().get_device());

    size_t embedding_size = tensor_shape[3];
    auto out_tensors = tt::operations::primary::moreh_layernorm(
        tensor->get_value(),
        embedding_size,
        1e-4F,
        /* gamma */ gamma->get_value(),
        /* beta */ beta->get_value(),
        mean,
        rstd);

    out->set_value(out_tensors[0].value());

    autograd::GradFunction grad = [tensor, out, mean, rstd, gamma, beta, embedding_size]() {
        auto input_grad = core::zeros_like(tensor->get_value());
        auto gamma_grad = core::zeros_like(gamma->get_value());
        auto beta_grad = core::zeros_like(beta->get_value());

        auto res = tt::operations::primary::moreh_layernorm_backward(
            out->get_grad(),
            tensor->get_value(),
            mean,
            rstd,
            embedding_size,
            gamma->get_value(),
            input_grad,
            gamma_grad,
            beta_grad);

        tensor->add_grad(res[0].value());
        gamma->add_grad(res[1].value());
        beta->add_grad(res[2].value());
    };

    std::vector<autograd::NodeId> links = autograd::get_links(tensor);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}
}  // namespace ttml::ttml::ops