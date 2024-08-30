#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace ttml::tokenizers {

class TokenizerBase {
   public:
    // Pure virtual function to encode a string into a vector of token IDs
    virtual std::vector<int> encode(const std::string& text) const = 0;

    // Pure virtual function to decode a vector of token IDs back into a string
    virtual std::string decode(const std::vector<int>& tokens) const = 0;

    // Virtual destructor for proper cleanup in derived classes
    virtual ~TokenizerBase() {}
};
}  // namespace ttml::tokenizers