#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <vector>

#include "serialization/hdf5_file.hpp"

namespace fs = std::filesystem;

class HDF5FileTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_filename = "/tmp/test_file.h5";
        // Ensure the test file does not exist before each test
        if (fs::exists(test_filename)) {
            fs::remove(test_filename);
        }
    }

    void TearDown() override {
        // Clean up after each test
        if (fs::exists(test_filename)) {
            fs::remove(test_filename);
        }
    }

    std::string test_filename;
};

TEST_F(HDF5FileTest, CreateAndOpenFile) {
    // Test creating a new HDF5 file
    ttml::serialization::HDF5File file(test_filename, false);
    EXPECT_TRUE(fs::exists(test_filename));
}

TEST_F(HDF5FileTest, CreateStorage) {
    ttml::serialization::HDF5File file(test_filename, false);

    std::vector<unsigned long long> dims = {10};
    file.create_storage<int>("dataset_", dims);
    // No exception means success
}

TEST_F(HDF5FileTest, WriteAndReadStorage) {
    ttml::serialization::HDF5File file(test_filename, false);

    std::vector<unsigned long long> dims = {5};
    file.create_storage<float>("dataset_float", dims);

    std::vector<float> data = {1.1F, 2.2F, 3.3F, 4.4F, 5.5F};
    file.write_storage("dataset_float", std::span(data));

    // Re-open the file in read-only mode
    ttml::serialization::HDF5File file_read(test_filename, true);
    std::vector<float> read_data = file_read.read_storage<float>("dataset_float");

    EXPECT_EQ(data.size(), read_data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_DOUBLE_EQ(data[i], read_data[i]);
    }
}

TEST_F(HDF5FileTest, WriteAndReadAttribute) {
    ttml::serialization::HDF5File file(test_filename, false);

    std::vector<unsigned long long> dims = {1};
    file.create_storage<int>("dataset_int", dims);

    int attr_value = 42;
    file.write_attribute("dataset_int", "attribute_int", attr_value);

    // Re-open the file in read-only mode
    ttml::serialization::HDF5File file_read(test_filename, true);
    int read_attr_value = file_read.read_attribute<int>("dataset_int", "attribute_int");

    EXPECT_EQ(attr_value, read_attr_value);
}

TEST_F(HDF5FileTest, WriteAndReadAttributeVector) {
    ttml::serialization::HDF5File file(test_filename, false);

    std::vector<unsigned long long> dims = {1};
    file.create_storage<int>("dataset_int", dims);

    std::vector<float> attr_values = {0.1F, 0.2F, 0.3F};
    file.write_attribute_vec("dataset_int", "attribute_vec", std::span(attr_values));

    // Re-open the file in read-only mode
    ttml::serialization::HDF5File file_read(test_filename, true);
    std::vector<float> read_attr_values = file_read.read_attribute_vec<float>("dataset_int", "attribute_vec");

    EXPECT_EQ(attr_values.size(), read_attr_values.size());
    for (size_t i = 0; i < attr_values.size(); ++i) {
        EXPECT_FLOAT_EQ(attr_values[i], read_attr_values[i]);
    }
}

TEST_F(HDF5FileTest, MoveConstructorAndAssignment) {
    ttml::serialization::HDF5File file(test_filename, false);

    // Move constructor
    ttml::serialization::HDF5File moved_file(std::move(file));
    EXPECT_TRUE(fs::exists(test_filename));

    // Move assignment
    ttml::serialization::HDF5File another_file("another_test_file.h5", false);
    another_file = std::move(moved_file);
    EXPECT_TRUE(fs::exists(test_filename));
    fs::remove("another_test_file.h5");
}
