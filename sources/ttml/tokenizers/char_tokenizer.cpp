#include "char_tokenizer.hpp"

#include <sstream>
#include <stdexcept>

namespace ttml::tokenizers {
CharTokenizer::CharTokenizer(const Vocabulary& vocabulary) : m_vocabulary(vocabulary) { build_reverse_mapping(); }

std::vector<int> CharTokenizer::encode(const std::string& text) const {
    std::vector<int> tokens;
    for (char c : text) {
        auto it = m_vocabulary.find(c);
        if (it != m_vocabulary.end()) {
            tokens.push_back(it->second);
        } else {
            throw std::runtime_error("Character not in vocabulary: " + std::string(1, c));
        }
    }
    return tokens;
}

std::string CharTokenizer::decode(const std::vector<int>& tokens) const {
    std::ostringstream oss;
    for (int token : tokens) {
        auto it = m_id_to_char.find(token);
        if (it != m_id_to_char.end()) {
            oss << it->second;
        } else {
            throw std::runtime_error("Token ID not in reverse vocabulary: " + std::to_string(token));
        }
    }
    return oss.str();
}
const CharTokenizer::Vocabulary& CharTokenizer::get_vocabulary() const { return m_vocabulary; }

void CharTokenizer::build_reverse_mapping() {
    for (const auto& pair : m_vocabulary) {
        m_id_to_char[pair.second] = pair.first;
    }
}
}  // namespace ttml::tokenizers