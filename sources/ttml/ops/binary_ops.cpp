#include "binary_ops.hpp"

#include <memory>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/eltwise/binary_backward/binary_backward.hpp>
#include <ttnn/tensor/types.hpp>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/graph.hpp"
#include "autograd/tensor.hpp"
#include "core/ttnn_all_includes.hpp"

namespace ttml::ops {

autograd::TensorPtr operator+(const autograd::TensorPtr& a, const autograd::TensorPtr& b) {
    autograd::TensorPtr out = std::make_shared<autograd::Tensor>();

    out->set_value(ttnn::add(a->get_value(), b->get_value()));
    autograd::GradFunction grad = [a, b, out]() {
        tt::tt_metal::MemoryConfig mem_config;

        auto res = ttnn::add_bw(out->get_grad(), a->get_value(), b->get_value(), mem_config);

        a->add_grad(res[0]);
        b->add_grad(res[1]);
    };
    std::vector<autograd::NodeId> links;

    const auto& a_node = a->get_node();
    const auto& b_node = b->get_node();
    if (a_node.has_value()) {
        links.push_back(a_node.value());
    }

    if (b_node.has_value()) {
        links.push_back(b_node.value());
    }
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}

autograd::TensorPtr operator-(const autograd::TensorPtr& a, const autograd::TensorPtr& b) {
    autograd::TensorPtr out = std::make_shared<autograd::Tensor>();

    out->set_value(ttnn::subtract(a->get_value(), b->get_value()));
    autograd::GradFunction grad = [a, b, out]() {
        tt::tt_metal::MemoryConfig mem_config;

        auto res = ttnn::sub_bw(out->get_grad(), a->get_value(), b->get_value(), mem_config);

        a->add_grad(res[0]);
        b->add_grad(res[1]);
    };
    std::vector<autograd::NodeId> links;

    const auto& a_node = a->get_node();
    const auto& b_node = b->get_node();
    if (a_node.has_value()) {
        links.push_back(a_node.value());
    }

    if (b_node.has_value()) {
        links.push_back(b_node.value());
    }
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}

autograd::TensorPtr operator*(const autograd::TensorPtr& a, const autograd::TensorPtr& b) {
    autograd::TensorPtr out = std::make_shared<autograd::Tensor>();

    out->set_value(ttnn::multiply(a->get_value(), b->get_value()));
    autograd::GradFunction grad = [a, b, out]() {
        tt::tt_metal::MemoryConfig mem_config;

        auto res = ttnn::mul_bw(0, out->get_grad(), a->get_value(), b->get_value(), mem_config);

        a->add_grad(res[0].value());
        b->add_grad(res[1].value());
    };
    std::vector<autograd::NodeId> links;

    const auto& a_node = a->get_node();
    const auto& b_node = b->get_node();
    if (a_node.has_value()) {
        links.push_back(a_node.value());
    }

    if (b_node.has_value()) {
        links.push_back(b_node.value());
    }
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}

autograd::TensorPtr operator/(const autograd::TensorPtr& a, const autograd::TensorPtr& b) {
    autograd::TensorPtr out = std::make_shared<autograd::Tensor>();
    out->set_value(ttnn::divide(a->get_value(), b->get_value()));
    autograd::GradFunction grad = [a, b, out]() {
        tt::tt_metal::MemoryConfig mem_config;

        auto res = ttnn::div_bw(out->get_grad(), a->get_value(), b->get_value(), "None", mem_config);

        a->add_grad(res[0]);
        b->add_grad(res[1]);
    };
    std::vector<autograd::NodeId> links;

    const auto& a_node = a->get_node();
    const auto& b_node = b->get_node();
    if (a_node.has_value()) {
        links.push_back(a_node.value());
    }

    if (b_node.has_value()) {
        links.push_back(b_node.value());
    }
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}

autograd::TensorPtr add(const autograd::TensorPtr& a, const autograd::TensorPtr& b) { return a + b; }

autograd::TensorPtr sub(const autograd::TensorPtr& a, const autograd::TensorPtr& b) { return a - b; }

autograd::TensorPtr mul(const autograd::TensorPtr& a, const autograd::TensorPtr& b) { return a * b; }

autograd::TensorPtr div(const autograd::TensorPtr& a, const autograd::TensorPtr& b) { return a / b; }

}  // namespace ttml::ops
