#include <cstdint>

#include "autograd/tensor.hpp"
#include "dropout_module.hpp"
#include "linear_module.hpp"
#include "ops/scaled_dot_product_attention.hpp"

namespace ttml::modules {

class MultiHeadAttention : public ttml::autograd::ModuleBase {
    uint32_t embedding_dim;
    uint32_t num_heads;
    std::shared_ptr<LinearLayer> q_linear;
    std::shared_ptr<LinearLayer> k_linear;
    std::shared_ptr<LinearLayer> v_linear;
    std::shared_ptr<LinearLayer> out_linear;
    std::shared_ptr<DropoutLayer> dropout;

public:
    explicit MultiHeadAttention(uint32_t embedding_dim, uint32_t num_heads, float dropout_prob);

    autograd::TensorPtr operator()(const autograd::TensorPtr& x, const autograd::TensorPtr& mask);
};

}  // namespace ttml::modules