// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "graph.hpp"

namespace ttml::autograd {

const std::vector<std::vector<size_t>>& Graph::get_edges() const {
    return m_links;
}

const std::vector<GraphNode>& Graph::get_graph_nodes() const {
    return m_graph_nodes;
}

NodeId Graph::add_node(GradFunction&& grad_function, std::span<NodeId> links) {
    size_t curr_id = m_graph_nodes.size();
    m_graph_nodes.emplace_back(std::move(grad_function));

    auto& node_links = m_links.emplace_back();
    node_links.reserve(links.size());
    for (const auto& link : links) {
        node_links.push_back(link.get_id());
    }

    return {curr_id, this};
}

NodeId::NodeId(size_t node_id, Graph* graph) : m_node_id(node_id), m_graph(graph) {
}

size_t NodeId::get_id() const {
    return m_node_id;
}

Graph& NodeId::get_graph() const {
    return *m_graph;
}

void Graph::reset() {
    m_graph_nodes.clear();
    m_links.clear();
}
}  // namespace ttml::autograd
