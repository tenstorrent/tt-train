#include "utils.hpp"

#include "ops/unary_ops.hpp"

void LossAverageMeter::update(float loss, size_t count) {
    m_sum += loss * static_cast<float>(count);
    m_count += count;
}

float LossAverageMeter::average() const {
    if (m_count == 0) {
        return 0.F;
    }
    return m_sum / static_cast<float>(m_count);
}

void LossAverageMeter::reset() {
    m_sum = 0.0F;
    m_count = 0;
}

void Timers::start(const std::string_view& name) {
    m_timers[std::string(name)] = std::chrono::high_resolution_clock::now();
}

long long Timers::stop(const std::string_view& name) {
    auto start_time = m_timers.at(std::string(name));
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    return duration.count();
}

MNISTModel::MNISTModel() {
    m_fc1 = std::make_shared<ttml::modules::LinearLayer>(784, 128);
    m_fc2 = std::make_shared<ttml::modules::LinearLayer>(128, 64);
    m_fc3 = std::make_shared<ttml::modules::LinearLayer>(64, 10);
    m_dropout = std::make_shared<ttml::modules::DropoutLayer>(0.2F);

    m_layernorm1 = std::make_shared<ttml::modules::LayerNormLayer>(128);
    m_layernorm2 = std::make_shared<ttml::modules::LayerNormLayer>(10);

    create_name("MNISTModel");

    register_module(m_fc1, "fc1");
    register_module(m_fc2, "fc2");
    register_module(m_fc3, "fc3");
    register_module(m_dropout, "dropout");
    register_module(m_layernorm1, "layernorm1");
    register_module(m_layernorm2, "layernorm2");
}

ttml::autograd::TensorPtr MNISTModel::operator()(ttml::autograd::TensorPtr x) {
    x = (*m_dropout)(x);
    x = (*m_fc1)(x);
    x = (*m_layernorm1)(x);
    x = ttml::ops::relu(x);
    x = (*m_fc2)(x);
    x = (*m_layernorm2)(x);
    x = ttml::ops::relu(x);
    x = (*m_fc3)(x);
    return x;
}
