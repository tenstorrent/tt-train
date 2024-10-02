
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/device.hpp"
#include "core/tt_tensor_utils.hpp"
#include "core/ttnn_all_includes.hpp"
#include "serialization/msgpack_file.hpp"
#include "serialization/serialization.hpp"

class TensorFileTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Remove test file if it exists
        if (std::filesystem::exists(test_filename)) {
            std::filesystem::remove(test_filename);
        }
    }

    void TearDown() override {
        // Clean up test file after each test
        if (std::filesystem::exists(test_filename)) {
            std::filesystem::remove(test_filename);
        }
    }

    const std::string test_filename = "/tmp/test_tensor.msgpack";
};

TEST_F(TensorFileTest, SerializeDeserializeTensor) {
    ttml::serialization::MsgPackFile serializer;
    auto* device = &ttml::autograd::ctx().get_device();
    auto shape = ttml::core::create_shape({1, 2, 32, 321});
    auto tensor_zeros = ttml::core::zeros(shape, device);
    auto tensor_ones = ttml::core::ones(shape, device);

    // Write tensor to file
    ttml::serialization::write_ttnn_tensor(serializer, "tensor", tensor_ones);
    serializer.serialize(test_filename);
    ttml::serialization::MsgPackFile deserializer;
    deserializer.deserialize(test_filename);

    // Read tensor from file
    tt::tt_metal::Tensor tensor_read = tensor_zeros;
    ttml::serialization::read_ttnn_tensor(deserializer, "tensor", tensor_read);

    auto read_vec = ttml::core::to_vector(tensor_read);

    for (auto& val : read_vec) {
        EXPECT_EQ(val, 1.F);
    }
}