#include "losses.hpp"

#include "ops/binary_ops.hpp"
#include "ops/unary_ops.hpp"

namespace ttml::ops {

autograd::TensorPtr mse_loss(
    const autograd::TensorPtr& target, const autograd::TensorPtr& prediction, ReduceType reduce) {
    auto diff = ops::sub(target, prediction);
    auto diff_2 = ops::mul(diff, diff);
    if (reduce == ReduceType::MEAN) {
        return ops::mean(diff_2);
    } else if (reduce == ReduceType::SUM) {
        return ops::sum(diff_2);
    } else {
        throw std::logic_error("Unsupported MSE type reduction type");
    }
}

}  // namespace ttml::ops