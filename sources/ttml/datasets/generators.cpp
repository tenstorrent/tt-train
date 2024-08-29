#include "generators.hpp"

namespace ttml::datasets {
InMemoryFloatVecDataset make_regression(MakeRegressionParams params, unsigned int seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0, 1.0);

    std::vector<std::vector<float>> data(params.n_samples, std::vector<float>(params.n_features));
    std::vector<std::vector<float>> targets(
        params.n_samples, std::vector<float>(params.n_targets));  // Targets are vectors of size n_targets

    // Generate random coefficients for each target
    std::vector<std::vector<float>> coefficients(params.n_targets, std::vector<float>(params.n_features));
    for (size_t t = 0; t < params.n_targets; ++t) {
        for (size_t i = 0; i < params.n_features; ++i) {
            coefficients[t][i] = dist(gen);
        }
    }

    for (size_t i = 0; i < params.n_samples; ++i) {
        for (size_t t = 0; t < params.n_targets; ++t) {
            float target = 0.0f;
            for (size_t j = 0; j < params.n_features; ++j) {
                data[i][j] = dist(gen);
                target += data[i][j] * coefficients[t][j];
            }

            if (params.bias) {
                target += dist(gen);  // Bias term
            }

            target += params.noise * dist(gen);  // Add noise
            targets[i][t] = target;              // Store the target in the target vector
        }
    }
    return {data, targets};
}
}  // namespace ttml::datasets