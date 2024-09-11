#pragma once
#include "core/template_utils.hpp"
#include "graph.hpp"

namespace ttml::autograd {

template <typename... Tensors>
std::vector<NodeId> get_links(Tensors&&... tensors) {
    static_assert(core::are_same_type<Tensors...>(), "All nodes must have the same type!");

    std::vector<NodeId> links;

    auto process_node = [&links](auto&& tensor) {
        const auto& node = tensor->get_node();
        if (node) {
            links.push_back(node.value());
        }
    };

    (process_node(std::forward<Tensors>(tensors)), ...);

    return links;
}
}  // namespace ttml::autograd