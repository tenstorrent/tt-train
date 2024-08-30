#pragma once
#include "char_tokenizer.hpp"

namespace ttml::tokenizers {

// right now it is very simple
class CharTokenizerTrainer {
   public:
    CharTokenizer train(const std::string& text, int starting_index = 1) const;
};
}  // namespace ttml::tokenizers