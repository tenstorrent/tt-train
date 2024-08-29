#pragma once

#include <random>
#include <vector>

#include "in_memory_dataset.hpp"

namespace ttml::datasets {

using InMemoryFloatVecDataset = InMemoryDataset<std::vector<float>, std::vector<float>>;

struct MakeRegressionParams {
    size_t n_samples = 1;
    size_t n_features = 1;
    size_t n_targets = 1;
    float noise = 0.0;
    bool bias = true;
};
InMemoryFloatVecDataset make_regression(MakeRegressionParams params, unsigned int seed = std::random_device{}());
}  // namespace ttml::datasets