#include "in_memory_char_dataset.hpp"

namespace ttml::datasets {

InMemoryCharDataset::InMemoryCharDataset(const std::vector<int>& tokens, int seq_length) :
    m_tokens(tokens), m_seq_length(seq_length) {}

[[nodiscard]] size_t InMemoryCharDataset::get_size_impl() const { return m_tokens.size() / m_seq_length; }

[[nodiscard]] InMemoryCharDataset::Sample InMemoryCharDataset::get_item_impl(size_t index) const {
    size_t dataset_size = get_size_impl();
    if (index >= dataset_size) {
        throw std::out_of_range("Index out of range");
    }

    size_t start_pos = index * m_seq_length;

    std::span<const int> input_span(m_tokens.data() + start_pos, m_seq_length);

    std::span<const int> target_span(m_tokens.data() + start_pos + 1, m_seq_length);

    return {input_span, target_span};
}

}  // namespace ttml::datasets