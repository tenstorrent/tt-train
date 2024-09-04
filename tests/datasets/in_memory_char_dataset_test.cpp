#include "datasets/in_memory_char_dataset.hpp"

#include <gtest/gtest.h>

using namespace ttml::datasets;

// Test fixture for InMemoryCharDataset
class InMemoryCharDatasetTest : public ::testing::Test {
   protected:
    // Example tokens for testing
    std::vector<int> tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // Sequence length
    int seq_length = 3;

    // Create an instance of InMemoryCharDataset
    InMemoryCharDataset dataset = InMemoryCharDataset(tokens, seq_length);
};

// Test get_size_impl function
TEST_F(InMemoryCharDatasetTest, GetSize) {
    // Expected number of samples
    size_t expected_size = tokens.size() / seq_length;

    ASSERT_EQ(dataset.get_size(), expected_size);
}

// Test get_item_impl function for the first sample
TEST_F(InMemoryCharDatasetTest, GetItemFirstSample) {
    size_t index = 0;

    auto sample = dataset.get_item(index);

    // Expected input and target spans
    std::vector<int> expected_input = {1, 2, 3};
    std::vector<int> expected_target = {2, 3, 4};

    ASSERT_EQ(std::vector<int>(sample.first.begin(), sample.first.end()), expected_input);
    ASSERT_EQ(std::vector<int>(sample.second.begin(), sample.second.end()), expected_target);
}

// Test get_item_impl function for the second sample
TEST_F(InMemoryCharDatasetTest, GetItemSecondSample) {
    size_t index = 1;

    auto sample = dataset.get_item(index);

    // Expected input and target spans
    std::vector<int> expected_input = {4, 5, 6};
    std::vector<int> expected_target = {5, 6, 7};

    ASSERT_EQ(std::vector<int>(sample.first.begin(), sample.first.end()), expected_input);
    ASSERT_EQ(std::vector<int>(sample.second.begin(), sample.second.end()), expected_target);
}

// Test get_item_impl function for the last sample
TEST_F(InMemoryCharDatasetTest, GetItemLastSample) {
    size_t index = dataset.get_size() - 1;

    auto sample = dataset.get_item(index);

    // Expected input and target spans
    std::vector<int> expected_input = {7, 8, 9};
    std::vector<int> expected_target = {8, 9, 10};

    ASSERT_EQ(std::vector<int>(sample.first.begin(), sample.first.end()), expected_input);
    ASSERT_EQ(std::vector<int>(sample.second.begin(), sample.second.end()), expected_target);
}

// Test out of range error for get_item_impl function
TEST_F(InMemoryCharDatasetTest, GetItemOutOfRange) {
    size_t index = dataset.get_size();  // Index out of range
    auto test_throw_lambda = [&]() { auto _ = dataset.get_item(index); };
    EXPECT_THROW(test_throw_lambda(), std::out_of_range);
}