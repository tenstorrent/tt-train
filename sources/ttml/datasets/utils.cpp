// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "utils.hpp"

#include "datasets/in_memory_token_dataset.hpp"
#include "tokenizers/bpe_tokenizer.hpp"
#include "tokenizers/char_tokenizer_trainer.hpp"

namespace {
constexpr auto gpt2_tokenizer_file_name = "/gpt2-tokenizer.json";
}
namespace ttml::datasets {

template <>
std::tuple<InMemoryTokenDataset, tokenizers::CharTokenizer> create_in_memory_token_dataset<tokenizers::CharTokenizer>(
    const std::string &text, uint32_t seq_length) {
    tokenizers::CharTokenizer tokenizer = tokenizers::CharTokenizerTrainer::train(text);

    std::vector<uint32_t> tokenized_text = tokenizer.encode(text);

    return {InMemoryTokenDataset(tokenized_text, seq_length), std::move(tokenizer)};
}

template <>
std::tuple<InMemoryTokenDataset, tokenizers::BPETokenizer> create_in_memory_token_dataset<tokenizers::BPETokenizer>(
    const std::string &text, uint32_t seq_length) {
    auto json_file_path = std::string(TOKENIZERS_DATA_PATH) + gpt2_tokenizer_file_name;
    auto tokenizer = tokenizers::BPETokenizer(json_file_path);

    const std::vector<uint32_t> tokenized_text = tokenizer.encode(text);

    return {InMemoryTokenDataset(tokenized_text, seq_length), std::move(tokenizer)};
}

}  // namespace ttml::datasets
