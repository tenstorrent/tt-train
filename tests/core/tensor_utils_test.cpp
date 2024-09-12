

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/device.hpp"
#include "core/tt_tensor_utils.hpp"
#include "core/ttnn_all_includes.hpp"
#include "init/cpu_initializers.hpp"

TEST(TensorUtilsTest, TestToFromTensorEven) {
    std::vector<float> test_data = {1.F, 5.F, 10.F, 15.F};

    tt::tt_metal::Shape shape = {1, 1, 1, 4};
    auto tensor = ttml::core::from_vector(test_data, ttnn::Shape(shape), &ttml::autograd::ctx().get_device());

    auto vec_back = ttml::core::to_vector(tensor);

    ASSERT_EQ(vec_back.size(), test_data.size());
    for (size_t i = 0; i < test_data.size(); i++) {
        EXPECT_EQ(vec_back[i], test_data[i]);
    }
}

TEST(TensorUtilsTest, TestToFromTensorOdd) {
    std::vector<float> test_data = {30.F, 20.F, 2.F};

    tt::tt_metal::Shape shape = {1, 1, 1, 3};
    auto tensor = ttml::core::from_vector(test_data, ttnn::Shape(shape), &ttml::autograd::ctx().get_device());

    auto vec_back = ttml::core::to_vector(tensor);

    ASSERT_EQ(vec_back.size(), test_data.size());
    for (size_t i = 0; i < test_data.size(); i++) {
        EXPECT_EQ(vec_back[i], test_data[i]);
    }
}

TEST(TensorUtilsTest, TestToFromTensorLargeWithBatch) {
    std::vector<float> test_data;
    uint32_t batch_size = 16;
    uint32_t vec_size = 256 * batch_size;
    for (size_t i = 0; i < vec_size; i++) {
        test_data.push_back((float)i / 100.0F);
    }

    tt::tt_metal::Shape shape{batch_size, 1, 1, vec_size / batch_size};
    auto tensor = ttml::core::from_vector(test_data, ttnn::Shape(shape), &ttml::autograd::ctx().get_device());
    auto vec_back = ttml::core::to_vector(tensor);
    ASSERT_EQ(vec_back.size(), test_data.size());
    for (size_t i = 0; i < test_data.size(); i++) {
        EXPECT_NEAR(vec_back[i], test_data[i], 0.5F);
    }
}

TEST(TensorUtilsTest, TestToFromTensorLarge) {
    std::vector<float> test_data;
    uint32_t vec_size = 1337;
    for (size_t i = 0; i < vec_size; i++) {
        test_data.push_back((float)i / 100.0F);
    }

    tt::tt_metal::Shape shape{1, 1, 1, vec_size};
    auto tensor = ttml::core::from_vector(test_data, ttnn::Shape(shape), &ttml::autograd::ctx().get_device());
    auto vec_back = ttml::core::to_vector(tensor);
    ASSERT_EQ(vec_back.size(), test_data.size());
    for (size_t i = 0; i < test_data.size(); i++) {
        EXPECT_NEAR(vec_back[i], test_data[i], 0.1F);
    }
}

TEST(TensorUtilsTest, TestToFromTensorBatch) {
    std::vector<float> test_data = {1.F, 5.F, 10.F, 15.F};

    tt::tt_metal::Shape shape = {2, 1, 1, 2};
    auto tensor = ttml::core::from_vector(test_data, ttnn::Shape(shape), &ttml::autograd::ctx().get_device());

    auto vec_back = ttml::core::to_vector(tensor);

    ASSERT_EQ(vec_back.size(), test_data.size());
    for (size_t i = 0; i < test_data.size(); i++) {
        EXPECT_EQ(vec_back[i], test_data[i]);
    }
}

TEST(TensorUtilsTest, TestToFromTensorMNIST) {
    std::vector<float> test_data;
    uint32_t mnist_size = 784;
    uint32_t vec_size = 10 * mnist_size;
    for (size_t i = 0; i < vec_size; i++) {
        test_data.push_back((float)i / 1000.0F);
    }

    tt::tt_metal::Shape shape{1, 1, 10, mnist_size};
    auto tensor = ttml::core::from_vector(test_data, ttnn::Shape(shape), &ttml::autograd::ctx().get_device());
    auto vec_back = ttml::core::to_vector(tensor);
    ASSERT_EQ(vec_back.size(), test_data.size());
    for (size_t i = 0; i < test_data.size(); i++) {
        EXPECT_NEAR(vec_back[i], test_data[i], 0.1F);
    }
}

TEST(TensorUtilsTest, TestToFromTensorMNISTInit) {
    std::vector<float> test_data;
    uint32_t mnist_size = 784;
    uint32_t vec_size = 10 * mnist_size;
    test_data.resize(vec_size);
    ttml::init::xavier_normal_init(test_data, {10, mnist_size});

    tt::tt_metal::Shape shape{1, 1, 10, mnist_size};
    auto tensor = ttml::core::from_vector(test_data, ttnn::Shape(shape), &ttml::autograd::ctx().get_device());
    auto vec_back = ttml::core::to_vector(tensor);
    ASSERT_EQ(vec_back.size(), test_data.size());
    for (size_t i = 0; i < test_data.size(); i++) {
        EXPECT_NEAR(vec_back[i], test_data[i], 0.1F);
    }
}