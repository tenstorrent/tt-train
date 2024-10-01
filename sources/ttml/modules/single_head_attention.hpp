#include "autograd/tensor.hpp"
#include "dropout_module.hpp"
#include "linear_module.hpp"
#include "ops/scaled_dot_product_attention.hpp"

namespace ttml::modules {

class SingleHeadAttention : public ttml::autograd::ModuleBase {
    std::shared_ptr<LinearLayer> q_linear;
    std::shared_ptr<LinearLayer> k_linear;
    std::shared_ptr<LinearLayer> v_linear;
    std::shared_ptr<LinearLayer> out_linear;
    std::shared_ptr<DropoutLayer> dropout;

public:
    explicit SingleHeadAttention(uint32_t embedding_dim, float dropout_prob);

    autograd::TensorPtr operator()(const autograd::TensorPtr& x, const autograd::TensorPtr& mask);
};

}  // namespace ttml::modules