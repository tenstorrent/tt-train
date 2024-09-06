#include "graph.hpp"

namespace ttml::autograd {

const std::vector<std::vector<size_t>>& Graph::get_edges() { return m_links; }

const std::vector<GraphNode>& Graph::get_graph_nodes() { return m_graph_nodes; }

NodeId Graph::add_node(GradFunction&& grad_function, std::span<NodeId> links) {
    size_t curr_id = m_graph_nodes.size();
    m_graph_nodes.emplace_back(std::move(grad_function));
    std::vector<size_t> node_links;
    for (auto& link : links) {
        node_links.push_back(link.get_id());
    }
    m_links.push_back(std::move(node_links));
    return {curr_id, this};
}

NodeId::NodeId(size_t node_id, Graph* graph) : m_node_id(node_id), m_graph(graph) {}

size_t NodeId::get_id() const { return m_node_id; }

core::not_null<Graph*> NodeId::get_graph() const { return m_graph; }

}  // namespace ttml::autograd