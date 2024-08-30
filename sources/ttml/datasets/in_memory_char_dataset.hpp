#pragma once

#include <span>

#include "in_memory_dataset.hpp"

namespace ttml::datasets {
class InMemoryCharDataset : public DatasetBase<InMemoryCharDataset, std::span<const int>, std::span<const int>> {
   public:
    using Parent = DatasetBase<InMemoryCharDataset, std::span<const int>, std::span<const int>>;
    using Sample = typename Parent::Sample;
    friend class DatasetBase<InMemoryCharDataset, std::span<const int>, std::span<const int>>;
    ;

    InMemoryCharDataset(const std::vector<int>& tokens, int seq_length);

    InMemoryCharDataset(const InMemoryCharDataset&) = default;
    InMemoryCharDataset(InMemoryCharDataset&&) = default;
    ~InMemoryCharDataset() = default;

   private:
    [[nodiscard]] size_t get_size_impl() const;

    [[nodiscard]] Sample get_item_impl(size_t index) const;

    std::vector<int> m_tokens;
    int m_seq_length = 0;
};
}  // namespace ttml::datasets