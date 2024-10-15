// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "autograd/auto_context.hpp"
#include "autograd/module_base.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "core/ttnn_all_includes.hpp"
#include "init/cpu_initializers.hpp"

namespace ttml::modules {

class Embedding : public autograd::ModuleBase {
    autograd::TensorPtr m_weight;

    void initialize_tensors(uint32_t num_embeddings, uint32_t embedding_dim);

public:
    Embedding(uint32_t num_embeddings, uint32_t embedding_dim);

    [[nodiscard]] autograd::TensorPtr operator()(const autograd::TensorPtr& tensor);
};

}  // namespace ttml::modules