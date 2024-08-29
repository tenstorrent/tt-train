#include "dataset_base.hpp"

namespace ttml::datasets {
template <class DataType, class TargetType>
class InMemoryDataset : public DatasetBase<InMemoryDataset<DataType, TargetType>, DataType, TargetType> {
   public:
    using Parent = DatasetBase<InMemoryDataset, DataType, TargetType>;
    using Sample = typename Parent::Sample;
    InMemoryDataset(const std::vector<DataType>& data, const std::vector<TargetType>& targets) :
        m_data(data), m_targets(targets) {}

    InMemoryDataset(const InMemoryDataset&) = default;
    InMemoryDataset(InMemoryDataset&&) = default;
    ~InMemoryDataset() = default;

    [[nodiscard]] size_t get_size_impl() const { return m_data.size(); }

    [[nodiscard]] Sample get_item_impl(size_t index) const { return {m_data[index], m_targets[index]}; }

   private:
    std::vector<DataType> m_data;
    std::vector<TargetType> m_targets;
};
}  // namespace ttml::datasets