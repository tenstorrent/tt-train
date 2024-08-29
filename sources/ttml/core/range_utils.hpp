#pragma once

namespace ttnn::core {
template <std::ranges::range R>
auto to_vector(R&& r) {
    auto r_common = r | std::views::common;
    return std::vector(r_common.begin(), r_common.end());
}
}  // namespace ttnn::core