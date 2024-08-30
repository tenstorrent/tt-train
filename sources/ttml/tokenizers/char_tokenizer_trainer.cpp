#include "char_tokenizer_trainer.hpp"

#include <algorithm>
#include <set>
#include <string>

namespace ttml::tokenizers {
CharTokenizer CharTokenizerTrainer::train(const std::string& text, int starting_index) const {
    CharTokenizer::Vocabulary vocabulary;

    // using set instead of unordered_set to stabilize order
    std::set<char> unique_chars(text.begin(), text.end());

    int id = starting_index;
    for (char c : unique_chars) {
        vocabulary[c] = id++;
    }

    return CharTokenizer(vocabulary);
}
}  // namespace ttml::tokenizers