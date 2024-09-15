#include "layernorm_op.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <ttnn/deprecated/tt_dnn/op_library/moreh_layernorm/moreh_layernorm_op.hpp>
#include <ttnn/deprecated/tt_dnn/op_library/moreh_layernorm_backward/moreh_layernorm_backward_op.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/eltwise/unary/unary.hpp>
#include <ttnn/tensor/tensor.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/graph.hpp"
#include "autograd/graph_utils.hpp"
#include "core/tt_tensor_utils.hpp"
#include "core/ttnn_all_includes.hpp"

namespace ttml::ttml::ops {
/*
std::vector<std::optional<Tensor>> moreh_layernorm(
    const Tensor &input,
    uint32_t normalized_dims,
    float eps,
    const std::optional<const Tensor> gamma = std::nullopt,
    const std::optional<const Tensor> beta = std::nullopt,
    const std::optional<const Tensor> output = std::nullopt,
    const std::optional<const Tensor> mean = std::nullopt,
    const std::optional<const Tensor> rstd = std::nullopt,
    const std::optional<MemoryConfig> &memory_config = std::nullopt,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt
    );


Tensor moreh_layernorm_backward_input_grad(
    const Tensor &output_grad,
    const Tensor &input,
    const Tensor &mean,
    const Tensor &rstd,
    uint32_t normalized_dims,
    const std::optional<const Tensor> input_grad = std::nullopt,
    const std::optional<const Tensor> gamma = std::nullopt,
    const std::optional<MemoryConfig> &memory_config = std::nullopt,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

*/
autograd::TensorPtr layernorm(const autograd::TensorPtr& tensor) {
    autograd::TensorPtr out = std::make_shared<autograd::Tensor>();
    tt::tt_metal::Tensor mean = core::zeros(
        ttnn::Shape(std::array<uint32_t, 4>{tensor->get_value().get_shape()[0], 1, 1, 1}),
        &autograd::ctx().get_device());
    tt::tt_metal::Tensor rstd = core::zeros(
        ttnn::Shape(std::array<uint32_t, 4>{tensor->get_value().get_shape()[0], 1, 1, 1}),
        &autograd::ctx().get_device());

    auto out_tensors = tt::operations::primary::moreh_layernorm(
        tensor->get_value(), 0, 1e-4F, /*beta*/ std::nullopt, /*gamma*/ std::nullopt, mean, rstd);

    out->set_value(out_tensors[0].value());

    autograd::GradFunction grad = [tensor, out, mean, rstd]() {
        auto res = tt::operations::primary::moreh_layernorm_backward_input_grad(
            out->get_grad(), tensor->get_value(), mean, rstd, 0);
        tensor->add_grad(res);
    };

    std::vector<autograd::NodeId> links = autograd::get_links(tensor);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}
}  // namespace ttml::ttml::ops