// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <span>

#include "dataset_base.hpp"

namespace ttml::datasets {
class InMemoryCharDataset
    : public DatasetBase<InMemoryCharDataset, std::span<const uint32_t>, std::span<const uint32_t>> {
public:
    using Parent = DatasetBase<InMemoryCharDataset, std::span<const uint32_t>, std::span<const uint32_t>>;
    using Sample = typename Parent::Sample;
    friend Parent;

    InMemoryCharDataset(const std::vector<uint32_t>& tokens, uint32_t seq_length);

    InMemoryCharDataset(const InMemoryCharDataset&) = default;
    InMemoryCharDataset(InMemoryCharDataset&&) = default;
    InMemoryCharDataset& operator=(const InMemoryCharDataset&) = default;
    InMemoryCharDataset& operator=(InMemoryCharDataset&&) = default;
    ~InMemoryCharDataset() = default;

private:
    [[nodiscard]] size_t get_size_impl() const;

    [[nodiscard]] Sample get_item_impl(size_t index) const;

    std::vector<uint32_t> m_tokens;
    uint32_t m_seq_length = 0;
};
}  // namespace ttml::datasets