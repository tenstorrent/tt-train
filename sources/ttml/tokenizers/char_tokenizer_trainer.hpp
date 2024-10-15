// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "char_tokenizer.hpp"

namespace ttml::tokenizers {

// right now it is very simple
class CharTokenizerTrainer {
public:
    [[nodiscard]] static CharTokenizer train(const std::string& text, bool add_padding_token = true);
};
}  // namespace ttml::tokenizers
