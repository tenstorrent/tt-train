#include "cpu_initializers.hpp"

#include <random>

#include "autograd/auto_context.hpp"
#include "fmt/core.h"

namespace ttml::init {

void uniform_init(std::vector<float>& vec, UniformRange range) {
    auto& [a, b] = range;

    std::uniform_real_distribution<float> dist(a, b);

    std::generate(
        vec.begin(), vec.end(), [&]() { return dist(autograd::AutoContext::get_instance().get_generator()); });
}

void normal_init(std::vector<float>& vec, NormalParams params) {
    auto& [mean, stddev] = params;

    std::normal_distribution<float> dist(mean, stddev);

    std::generate(
        vec.begin(), vec.end(), [&]() { return dist(autograd::AutoContext::get_instance().get_generator()); });
}

void constant_init(std::vector<float>& vec, float value) {
    // Fill the vector with the specified constant value
    std::fill(vec.begin(), vec.end(), value);
}

void xavier_uniform_init(std::vector<float>& vec, FanParams params) {
    auto& [fan_in, fan_out] = params;
    float limit = std::sqrt(6.0F / (float)(fan_in + fan_out));

    std::uniform_real_distribution<float> dist(-limit, limit);

    // Fill the vector with uniformly distributed random values in the range [-limit, limit]
    std::generate(
        vec.begin(), vec.end(), [&]() { return dist(autograd::AutoContext::get_instance().get_generator()); });
}

void xavier_normal_init(std::vector<float>& vec, FanParams params) {
    auto& [fan_in, fan_out] = params;
    float stddev = std::sqrtf(2.0F / (float)(fan_in + fan_out));

    // Random number generator with a seed
    // Mersenne Twister generator
    std::normal_distribution<float> dist(0.0F, stddev);
    // auto& gen = autograd::AutoContext::get_instance().get_generator();

    std::generate(
        vec.begin(), vec.end(), [&]() { return dist(autograd::AutoContext::get_instance().get_generator()); });
}

void kaiming_uniform_init(std::vector<float>& vec, int fan_in) {
    float limit = std::sqrt(3.0F / (float)fan_in);

    std::uniform_real_distribution<float> dist(-limit, limit);

    // Fill the vector with uniformly distributed random values in the range [-limit, limit]
    std::generate(
        vec.begin(), vec.end(), [&]() { return dist(autograd::AutoContext::get_instance().get_generator()); });
}

void kaiming_normal_init(std::vector<float>& vec, int fan_out) {
    float stddev = std::sqrt(2.0F / (float)fan_out);

    std::normal_distribution<float> dist(0.0F, stddev);

    std::generate(
        vec.begin(), vec.end(), [&]() { return dist(autograd::AutoContext::get_instance().get_generator()); });
}

}  // namespace ttml::init