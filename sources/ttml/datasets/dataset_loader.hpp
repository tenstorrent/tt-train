#pragma once

namespace ttml::datasets {
#include <algorithm>
#include <iterator>
#include <random>
#include <ranges>
#include <vector>

template <typename DatasetType>
class DataLoader {
   public:
    using Sample = typename DatasetType::Sample;

    DataLoader(
        DatasetType& dataset, size_t batch_size, bool shuffle = false, unsigned int seed = std::random_device{}()) :
        m_dataset(dataset), m_batch_size(batch_size), m_shuffle(shuffle), m_indices(dataset.size()), m_seed(seed) {
        std::iota(m_indices.begin(), m_indices.end(), 0);

        if (m_shuffle) {
            shuffle_indices();
        }
    }

    class Iterator {
       public:
        Iterator(DataLoader& data_loader, size_t start_index) :
            data_loader_(data_loader), current_index_(start_index) {}

        Iterator& operator++() {
            current_index_ += data_loader_.m_batch_size;
            return *this;
        }

        std::vector<Sample> operator*() const { return data_loader_.fetch_batch(current_index_); }

        bool operator!=(const Iterator& other) const { return current_index_ != other.current_index_; }

       private:
        DataLoader& data_loader_;
        size_t current_index_;
    };

    Iterator begin() { return Iterator(*this, 0); }

    Iterator end() { return Iterator(*this, m_indices.size()); }

   private:
    DatasetType& m_dataset;
    size_t m_batch_size = 0;
    bool m_shuffle = false;
    unsigned int m_seed = 0;
    std::vector<size_t> m_indices;

    void shuffle_indices() {
        std::mt19937 g(m_seed);
        std::shuffle(m_indices.begin(), m_indices.end(), g);
    }

    std::vector<Sample> fetch_batch(size_t start_index) const {
        std::vector<Sample> batch =
            m_indices | std::ranges::views::drop(start_index) | std::ranges::views::take(end_index - start_index) |
            std::ranges::views::transform([&](size_t i) { return m_dataset.get(i); }) | std::ranges::to<std::vector>();
        return batch;
    }
};

}  // namespace ttml::datasets