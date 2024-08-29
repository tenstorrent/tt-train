#pragma once
#include <algorithm>
#include <random>
#include <ranges>
#include <vector>

#include "core/range_utils.hpp"

namespace ttml::datasets {

template <typename DatasetType>
class DataLoader {
   public:
    using Sample = typename DatasetType::Sample;

    DataLoader(
        DatasetType& dataset, size_t batch_size, bool shuffle = false, unsigned int seed = std::random_device{}()) :
        m_dataset(dataset), m_batch_size(batch_size), m_shuffle(shuffle), m_indices(dataset.get_size()), m_seed(seed) {
        std::iota(m_indices.begin(), m_indices.end(), 0);
    }

    void shuffle_indices() {
        if (!m_shuffle) {
            return;
        }
        std::mt19937 g(m_seed);
        std::shuffle(m_indices.begin(), m_indices.end(), g);
    }

    class Iterator {
       public:
        Iterator(DataLoader& data_loader, size_t start_index) :
            m_data_loader(data_loader), m_current_mutex(start_index) {}

        Iterator& operator++() {
            m_current_mutex += m_data_loader.m_batch_size;
            return *this;
        }

        std::vector<Sample> operator*() const { return m_data_loader.fetch_batch(m_current_mutex); }

        bool operator!=(const Iterator& other) const { return m_current_mutex != other.m_current_mutex; }

       private:
        DataLoader& m_data_loader;
        size_t m_current_mutex = 0;
    };

    Iterator begin() {
        shuffle_indices();
        return Iterator(*this, 0);
    }

    Iterator end() { return Iterator(*this, m_indices.size()); }

   private:
    DatasetType& m_dataset;
    size_t m_batch_size = 0;
    bool m_shuffle = false;
    unsigned int m_seed = 0;
    std::vector<size_t> m_indices;

    std::vector<Sample> fetch_batch(size_t start_index) const {
        size_t end_index = std::min(start_index + m_batch_size, m_indices.size());
        std::vector<Sample> batch;
        for (size_t i = start_index; i < end_index; ++i) {
            batch.push_back(m_dataset.get_item(m_indices[i]));
        }
        return batch;
    }
};

}  // namespace ttml::datasets