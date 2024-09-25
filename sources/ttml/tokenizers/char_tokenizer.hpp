#pragma once

#include <unordered_map>

#include "tokenizer_base.hpp"

namespace ttml::tokenizers {

class CharTokenizer : public TokenizerBase {
public:
    using Vocabulary = std::unordered_map<std::string, int>;
    using IdtoChars = std::unordered_map<int, std::string>;
    // Constructor that initializes the tokenizer with a vocabulary
    explicit CharTokenizer(Vocabulary vocabulary);

    CharTokenizer(const CharTokenizer&) = default;
    CharTokenizer& operator=(const CharTokenizer&) = default;

    CharTokenizer(CharTokenizer&&) = default;
    CharTokenizer& operator=(CharTokenizer&&) = default;

    [[nodiscard]] std::vector<int> encode(const std::string& text) const override;

    [[nodiscard]] std::string decode(const std::vector<int>& tokens) const override;

    [[nodiscard]] const CharTokenizer::Vocabulary& get_vocabulary() const;

    ~CharTokenizer() override = default;

private:
    Vocabulary m_vocabulary;
    IdtoChars m_id_to_char;

    void build_reverse_mapping();
};

}  // namespace ttml::tokenizers