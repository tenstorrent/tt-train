#include "linear_op.hpp"

#include <tt_dnn/op_library/moreh_mean/moreh_mean_op.hpp>
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
        auto bias_shape = core::get_shape_without_padding(bias->get_value());
        auto bias_grad_tensor = core::zeros(ttnn::Shape(bias_shape), bias->get_value().device());
        tt::operations::primary::moreh_mean(
            out->get_grad(), /* dim */ 0, /* keepdim */ true, /* divisor */ std::nullopt, /* out */ bias_grad_tensor);
        bias->add_grad(bias_grad_tensor);
        weight->add_grad(ttnn::matmul(out->get_grad(), tensor->get_value(), /* transpose_a*/ true));
        tensor->add_grad(ttnn::matmul(out->get_grad(), weight->get_value()));
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