#include "losses.hpp"

#include "ops/binary_ops.hpp"
#include "ops/unary_ops.hpp"

namespace ttml::ops {

autograd::TensorPtr mse_loss(
    const autograd::TensorPtr& target, const autograd::TensorPtr& prediction, ReduceType reduce) {
    auto diff = ops::sub(target, prediction);  // TODO: @rfurko-tt use "ttnn::squared_difference"
    auto diff_2 = ops::mul(diff, diff);  // TODO: need to add backward "ttnn::squared_difference_bw" might be faster
    if (reduce == ReduceType::MEAN) {
        return ops::mean(diff_2);
    } else {
        throw std::logic_error("Unsupported MSE reduction type");
    }
}

}  // namespace ttml::ops