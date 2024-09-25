#include "char_tokenizer_trainer.hpp"

#include <algorithm>
#include <set>
#include <string>

namespace ttml::tokenizers {

CharTokenizer CharTokenizerTrainer::train(const std::string& text, bool add_padding_token) {
    CharTokenizer::Vocabulary vocabulary;

    // using set instead of unordered_set to stabilize order
    std::set<char> unique_chars(text.begin(), text.end());

    if (add_padding_token) {
        vocabulary["<PAD>"] = 0;
    }

    for (char chr : unique_chars) {
        vocabulary[std::string(1, chr)] = static_cast<int>(vocabulary.size());
    }

    return CharTokenizer(vocabulary);
}

}  // namespace ttml::tokenizers