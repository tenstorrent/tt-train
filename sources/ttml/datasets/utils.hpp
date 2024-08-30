#pragma once
#include "in_memory_char_dataset.hpp"
#include "tokenizers/char_tokenizer.hpp"

namespace ttml::datasets {

std::tuple<InMemoryCharDataset, tokenizers::CharTokenizer> create_in_memory_char_dataset(
    const std::string& text, int seq_length);
}