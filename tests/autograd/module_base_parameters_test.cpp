#include <gtest/gtest.h>

#include <autograd/auto_context.hpp>
#include <memory>

#include "autograd/module_base.hpp"
#include "modules/dropout_module.hpp"
#include "modules/layer_norm_module.hpp"
#include "modules/linear_module.hpp"
#include "ops/unary_ops.hpp"

class Model : public ttml::autograd::ModuleBase {
    std::shared_ptr<ttml::modules::LinearLayer> m_fc1;
    std::shared_ptr<ttml::modules::LinearLayer> m_fc2;

public:
    Model() {
        m_fc1 = std::make_shared<ttml::modules::LinearLayer>(784, 128);
        m_fc2 = std::make_shared<ttml::modules::LinearLayer>(128, 64);

        create_name("Model");

        register_module(m_fc1, "fc1");
        register_module(m_fc2, "fc2");
    }

    ttml::autograd::TensorPtr operator()(ttml::autograd::TensorPtr x) {
        x = (*m_fc1)(x);
        x = ttml::ops::relu(x);
        x = (*m_fc2)(x);
        return x;
    }
};

class ModuleBaseParametersTest : public ::testing::Test {
protected:
    void TearDown() override {
        ttml::autograd::ctx().reset_graph();
    }
};

TEST_F(ModuleBaseParametersTest, AllParametersIncluded) {
    Model model;
    auto model_params = model.parameters();
    // 2 LinearLayer modules: 2 weight tensors and 2 bias tensors
    EXPECT_EQ(model_params.size(), 4);
};