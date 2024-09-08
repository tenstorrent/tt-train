#pragma once

#include <ttnn/tensor/tensor.hpp>

#include "autograd/module_base.hpp"
#include "core/tt_tensor_utils.hpp"

namespace ttml::optimizers {

struct SGDConfig {
    float lr{1e-3F};
    float momentum{0.0F};
};

class SGD {
public:
    explicit SGD(ttml::autograd::NamedParameters parameters, const SGDConfig& config);

    void zero_grad();

    void step();

private:
    float m_lr{1e-3F};
    float m_momentum{0.0F};
    ttml::autograd::NamedParameters m_parameters;
    std::unordered_map<std::string, tt::tt_metal::Tensor> m_theta;
};

}  // namespace ttml::optimizers