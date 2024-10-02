#pragma once

#include <ttnn/tensor/tensor.hpp>

#include "autograd/module_base.hpp"
#include "core/tt_tensor_utils.hpp"

namespace ttml::optimizers {

using TTTensorDict = std::unordered_map<std::string, tt::tt_metal::Tensor>;

struct SGDConfig {
    float lr{1e-3F};
    float momentum{0.0F};
    float dampening{0.0F};
    float weight_decay{0.0F};
    bool nesterov{false};
};

class SGD {
public:
    explicit SGD(ttml::autograd::NamedParameters parameters, const SGDConfig& config);

    void zero_grad();

    void step();

    const TTTensorDict& get_theta() const;
    TTTensorDict& get_theta();

private:
    size_t steps{0};
    SGDConfig m_config;
    ttml::autograd::NamedParameters m_parameters;
    TTTensorDict m_theta;
};

}  // namespace ttml::optimizers