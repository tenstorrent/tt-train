#include "linear_module.hpp"

namespace ttml::modules {

// TODO: finish initialization
void LinearLayer::initialize_tensors([[maybe_unused]] uint32_t in_features, [[maybe_unused]] uint32_t out_features) {}

const std::string& LinearLayer::get_name() const { return m_name; }

LinearLayer::LinearLayer(uint32_t in_features, uint32_t out_features) {
    initialize_tensors(in_features, out_features);

    create_name("linear");
    register_tensor(m_weight, "weight");
    register_tensor(m_bias, "bias");
}

autograd::TensorPtr LinearLayer::operator()(const autograd::TensorPtr& tensor) {
    return ops::linear_op(tensor, m_weight, m_bias);
}

}  // namespace ttml::modules