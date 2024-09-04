#pragma once

#include <functional>
#include <span>

#include "core/not_null.hpp"
namespace ttml::autograd {
class Graph;
class GraphNode;

using GradFunction = std::function<void()>;

struct GraphNode {
    GradFunction grad_function;
};

class NodeId {
   public:
    NodeId(size_t node_id, Graph* graph) : m_node_id(node_id), m_graph(graph) {}

   private:
    size_t m_node_id = 0;
    core::not_null<Graph*> m_graph;

   public:
    [[nodiscard]] size_t get_id() const;
};

class Graph {
   private:
    std::vector<GraphNode> m_graph_nodes;
    std::vector<std::vector<size_t>> m_links;

   public:
    NodeId add_node(GradFunction&& grad_function, std::span<NodeId> links);
};

}  // namespace ttml::autograd