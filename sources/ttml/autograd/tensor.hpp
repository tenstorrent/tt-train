#pragma once
#include <memory>
#include <optional>

#include "autograd/graph.hpp"
#include "core/ttnn_all_includes.hpp"
#include "graph.hpp"

namespace ttml::autograd {

class Tensor : public std::enable_shared_from_this<Tensor> {
   private:
    tt::tt_metal::Tensor m_value;
    tt::tt_metal::Tensor m_grad;
    bool m_require_grad = true;
    std::optional<NodeId> m_node_id;

   public:
};

using TensorPtr = std::shared_ptr<Tensor>;
}  // namespace ttml::autograd