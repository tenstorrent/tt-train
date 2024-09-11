#include "linear_op.hpp"

#include <ttnn/tensor/types.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "core/ttnn_all_includes.hpp"

namespace ttml::ops {

autograd::TensorPtr linear_op(
    const autograd::TensorPtr& tensor, const autograd::TensorPtr& weight, const autograd::TensorPtr& bias) {
    autograd::TensorPtr out = std::make_shared<autograd::Tensor>();
    out->set_value(ttnn::linear(
        tensor->get_value(), weight->get_value(), bias->get_value(), /* transpose_a */ false, /* tranpose_b */ true));

    autograd::GradFunction grad = [weight, bias, tensor, out]() {
        auto bias_grad = core::zeros_like(bias->get_value());
        auto tensor_grad = core::zeros_like(tensor->get_value());
        auto weight_grad = core::zeros_like(weight->get_value());

        auto res = tt::operations::primary::moreh_linear_backward(
            out->get_grad(),
            tensor->get_value(),
            weight->get_value(),
            /* are required outputs */ {true, true, true},
            bias->get_value(),
            tensor_grad,
            weight_grad,
            bias_grad);

        tensor->add_grad(res[0].value());
        weight->add_grad(res[1].value());
        bias->add_grad(res[2].value());
    };

    std::vector<autograd::NodeId> links;
    if (weight->get_node().has_value()) {
        links.push_back(weight->get_node().value());
    }
    if (bias->get_node().has_value()) {
        links.push_back(bias->get_node().value());
    }
    if (tensor->get_node().has_value()) {
        links.push_back(tensor->get_node().value());
    }

    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));
    return out;
}

}  // namespace ttml::ops