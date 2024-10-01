#include "utils.hpp"

#include "datasets/in_memory_char_dataset.hpp"
#include "tokenizers/char_tokenizer_trainer.hpp"

namespace ttml::datasets {
std::tuple<InMemoryCharDataset, tokenizers::CharTokenizer> create_in_memory_char_dataset(
    const std::string& text, uint32_t seq_length) {
    tokenizers::CharTokenizer tokenizer = tokenizers::CharTokenizerTrainer::train(text);

    std::vector<uint32_t> tokenized_text = tokenizer.encode(text);

    return {InMemoryCharDataset(tokenized_text, seq_length), std::move(tokenizer)};
}
}  // namespace ttml::datasets