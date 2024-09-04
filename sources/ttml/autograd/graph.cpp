#include "graph.hpp"

namespace ttml::autograd {
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
}  // namespace ttml::autograd