#pragma once
#include <type_traits>
namespace ttml::core {
template <typename T, typename... Rest>
constexpr bool are_same_type() {
    return (std::is_same_v<std::decay_t<T>, std::decay_t<Rest>> && ...);
}
}  // namespace ttml::core