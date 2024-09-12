#include "dropout_module.hpp"

#include "autograd/module_base.hpp"
#include "ops/dropout_op.hpp"
namespace ttml::modules {

DropoutLayer::DropoutLayer(float probability) : m_prob(probability) {}

[[nodiscard]] autograd::TensorPtr DropoutLayer::operator()(const autograd::TensorPtr& tensor) {
    if (this->get_train_mode() == autograd::TrainMode::EVAL) {
        return tensor;
    }

    return ttml::ops::dropout(tensor, m_prob);
}
}  // namespace ttml::modules
