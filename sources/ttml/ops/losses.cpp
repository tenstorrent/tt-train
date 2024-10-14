#include "losses.hpp"

#include "autograd/auto_context.hpp"
#include "autograd/graph_utils.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/tt_tensor_utils.hpp"
#include "core/ttnn_all_includes.hpp"
#include "ops/binary_ops.hpp"
#include "ops/unary_ops.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

namespace ttml::ops {

autograd::TensorPtr mse_loss(
    const autograd::TensorPtr& prediction, const autograd::TensorPtr& target, ReduceType reduce) {
    auto difference = ops::sub(target, prediction);  // TODO: @rfurko-tt use "ttnn::squared_difference"
    auto squared_difference =
        ops::mul(difference, difference);  // TODO: need to add backward "ttnn::squared_difference_bw" might be faster
    if (reduce == ReduceType::MEAN) {
        return ops::mean(squared_difference);
    } else {
        throw std::logic_error("Unsupported MSE reduction type");
    }
}

autograd::TensorPtr cross_entropy_loss_without_reduce_(
    const autograd::TensorPtr& prediction, const autograd::TensorPtr& target) {
    const float eps = 1e-6F;
    auto prediction_tensor = ttnn_fixed::softmax(prediction->get_value(), 3);
    auto prediction_tensor_clipped = ttnn::clip(prediction_tensor, eps, 1.0F);
    auto loss = ttnn::multiply(target->get_value(), ttnn::log(prediction_tensor_clipped));
    loss = ttnn::neg(loss);
    loss = ttnn_fixed::sum_over_dim(loss, 3);
    auto out = autograd::create_tensor(loss);

    autograd::GradFunction grad = [target, prediction_tensor, prediction, out]() {
        auto grad = ttnn::subtract(prediction_tensor, target->get_value());
        grad = ttnn::multiply(grad, out->get_grad());
        prediction->add_grad(grad);
    };

    auto links = autograd::get_links(prediction);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}

autograd::TensorPtr cross_entropy_loss(
    const autograd::TensorPtr& prediction, const autograd::TensorPtr& target, ReduceType reduce) {
    auto loss = cross_entropy_loss_without_reduce_(prediction, target);
    if (reduce == ReduceType::MEAN) {
        return ops::mean(loss);
    } else {
        throw std::logic_error("Unsupported cross entropy reduction type");
    }
}

}  // namespace ttml::ops