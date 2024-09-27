#pragma once
#include <numeric>
#include <random>
#include <span>

#include "autograd/auto_context.hpp"
#include "dataset_subset.hpp"
#include "in_memory_char_dataset.hpp"
#include "tokenizers/char_tokenizer.hpp"
namespace ttml::datasets {

std::tuple<InMemoryCharDataset, tokenizers::CharTokenizer> create_in_memory_char_dataset(
    const std::string& text, uint32_t seq_length);

template <typename DatasetType>
std::vector<DatasetSubset<DatasetType>> random_split(
    const DatasetType& dataset, std::span<size_t> split_sizes, bool shuffle = true) {
    size_t total_size = std::accumulate(split_sizes.begin(), split_sizes.end(), 0ULL);
    if (total_size != dataset.get_size()) {
        throw std::invalid_argument("Total of split sizes must equal the size of the dataset.");
    }

    // Create indices and shuffle them
    std::vector<size_t> indices(dataset.get_size());
    std::iota(indices.begin(), indices.end(), 0);

    if (shuffle) {
        std::mt19937& gen = autograd::AutoContext::get_instance().get_generator();
        std::shuffle(indices.begin(), indices.end(), gen);
    }

    // Create the subsets
    std::vector<DatasetSubset<DatasetType>> subsets;
    auto current_iter = indices.begin();
    for (size_t size : split_sizes) {
        std::vector<size_t> subset_indices(current_iter, current_iter + (long)size);
        subsets.emplace_back(dataset, std::move(subset_indices));
        current_iter += (long)size;
    }

    return subsets;
}

}  // namespace ttml::datasets