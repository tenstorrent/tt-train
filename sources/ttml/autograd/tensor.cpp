#include "tensor.hpp"

#include "core/not_null.hpp"
#include "core/tt_tensor_utils.hpp"

namespace {

// TODO: implement stack based topological sort
void topological_sort(
    size_t node_id,
    const std::vector<std::vector<size_t>>& edges,
    std::unordered_set<size_t>& visited,
    std::vector<size_t>& sorted_nodes) {
    if (visited.contains(node_id)) {
        return;
    }
    visited.insert(node_id);
    for (const auto& next_node : edges[node_id]) {
        topological_sort(next_node, edges, visited, sorted_nodes);
    }
    sorted_nodes.push_back(node_id);
}

}  // namespace

namespace ttml::autograd {

Tensor::Tensor(tt::tt_metal::Tensor m_value, bool require_grad) :
    m_value(std::move(m_value)), m_require_grad(require_grad) {}

void Tensor::add_grad(const tt::tt_metal::Tensor& grad) {
    try_init_grad();
    m_grad = ttnn::add_(m_grad, grad);
}

void Tensor::backward() {
    if (!m_node_id.has_value()) {
        return;
    }
    std::vector<size_t> sorted_nodes;
    std::unordered_set<std::size_t> visited_nodes;
    auto& graph = m_node_id->get_graph();
    topological_sort(m_node_id->get_id(), graph.get_edges(), visited_nodes, sorted_nodes);

    const auto& graph_nodes = graph.get_graph_nodes();
    std::ranges::reverse(sorted_nodes);
    try_init_grad(true);
    for (const auto& node_id : sorted_nodes) {
        graph_nodes[node_id].grad_function();
    }
}

void Tensor::try_init_grad(bool init_ones) {
    if (this->get_grad().tensor_attributes == nullptr) {
        if (init_ones) {
            this->set_grad(ttml::core::ones_like(m_value));
        } else {
            this->set_grad(ttml::core::zeros_like(m_value));
        }
    }
}
}  // namespace ttml::autograd