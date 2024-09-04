#pragma once
#include <memory>

#include "core/ttnn_all_includes.hpp"
#include "graph.hpp"

namespace ttml::autograd {

class Tensor : public std::enable_shared_from_this<Tensor> {
   private:
    tt::tt_metal::Tensor m_value;
    tt::tt_metal::Tensor m_grad;
    bool require_grad = true;
    GraphNode* grad_node = nullptr;

   public:
};

}  // namespace ttml::autograd