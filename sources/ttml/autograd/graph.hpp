#pragma once

#include <functional>

namespace ttml::autograd {
using GradFunction = std::function<void()>;

struct GraphNode {
    GradFunction grad_function;
};

class Graph {
   private:
    std::vector<GraphNode> m_graph_nodes;
    std::vector<std::vector<size_t>> m_dependency;

   public:
    void add_node(GraphNode&& node) {}
};

}  // namespace ttml::autograd