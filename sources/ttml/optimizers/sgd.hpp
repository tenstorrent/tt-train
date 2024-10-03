#pragma once

#include <ttnn/tensor/tensor.hpp>

#include "autograd/module_base.hpp"
#include "core/tt_tensor_utils.hpp"

namespace ttml::optimizers {

// TODO: in future we will create unordered_map of variant<Autograd::Tensor, Tensor, Scalar> or something like this for
// a base type.
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

    // I'd like to return copy of the dict like we do in module
    // but I cannot replace values in side of the ttnn tensor in a easy way, right now we are using get/set state dict
    // TODO: think about it and move to the autograd::tensors
    [[nodiscard]] TTTensorDict get_state_dict() const;
    void set_state_dict(TTTensorDict dict);

    [[nodiscard]] size_t get_steps() const;
    void set_steps(size_t steps);

private:
    size_t steps{0};
    SGDConfig m_config;
    ttml::autograd::NamedParameters m_parameters;
    TTTensorDict m_theta;
};

}  // namespace ttml::optimizers