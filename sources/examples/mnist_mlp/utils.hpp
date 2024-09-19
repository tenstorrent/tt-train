#pragma once

#include <fmt/core.h>

#include <cassert>
#include <chrono>
#include <cstddef>
#include <string>
#include <unordered_map>

#include "autograd/module_base.hpp"
#include "modules/dropout_module.hpp"
#include "modules/layer_norm_module.hpp"
#include "modules/linear_module.hpp"

class LossAverageMeter {
    float m_sum = 0.0F;
    size_t m_count = 0;

public:
    void update(float loss, size_t count = 1);

    [[nodiscard]] float average() const;

    void reset();
};

class Timers {
public:
    void start(const std::string_view& name);

    long long stop(const std::string_view& name);

private:
    std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> m_timers;
};

class MNISTModel : public ttml::autograd::ModuleBase {
    std::shared_ptr<ttml::modules::LinearLayer> m_fc1;
    std::shared_ptr<ttml::modules::LinearLayer> m_fc2;
    std::shared_ptr<ttml::modules::LinearLayer> m_fc3;
    std::shared_ptr<ttml::modules::DropoutLayer> m_dropout;
    std::shared_ptr<ttml::modules::LayerNormLayer> m_layernorm1;
    std::shared_ptr<ttml::modules::LayerNormLayer> m_layernorm2;

public:
    MNISTModel();

    ttml::autograd::TensorPtr operator()(ttml::autograd::TensorPtr x);
};