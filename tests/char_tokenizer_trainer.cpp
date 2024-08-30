
#include "tokenizers/char_tokenizer_trainer.hpp"

#include <gtest/gtest.h>

using namespace ttml::tokenizers;

// Test fixture for CharTokenizerTrainer
class CharTokenizerTrainerTest : public ::testing::Test {
   protected:
    // Example CharTokenizerTrainer instance
    CharTokenizerTrainer trainer;
};

// Test that the trainer creates a tokenizer with the correct vocabulary
TEST_F(CharTokenizerTrainerTest, TrainVocabulary) {
    std::string text = "hello world";
    CharTokenizer tokenizer = trainer.train(text);

    CharTokenizer::Vocabulary expected_vocabulary = {
        {' ', 1}, {'d', 2}, {'e', 3}, {'h', 4}, {'l', 5}, {'o', 6}, {'r', 7}, {'w', 8}};

    // Verify that the generated vocabulary matches the expected one
    ASSERT_EQ(tokenizer.get_vocabulary().size(), expected_vocabulary.size());

    for (const auto& pair : expected_vocabulary) {
        auto it = tokenizer.get_vocabulary().find(pair.first);
        ASSERT_NE(it, tokenizer.get_vocabulary().end());
        ASSERT_EQ(it->second, pair.second);
    }
}

// Test that the trainer handles duplicate characters correctly
TEST_F(CharTokenizerTrainerTest, TrainWithDuplicateCharacters) {
    std::string text = "aaaabbbb";
    CharTokenizer tokenizer = trainer.train(text);

    CharTokenizer::Vocabulary expected_vocabulary = {{'a', 1}, {'b', 2}};

    // Verify that the generated vocabulary has no duplicates
    ASSERT_EQ(tokenizer.get_vocabulary().size(), expected_vocabulary.size());

    for (const auto& pair : expected_vocabulary) {
        auto it = tokenizer.get_vocabulary().find(pair.first);
        ASSERT_NE(it, tokenizer.get_vocabulary().end());
        ASSERT_EQ(it->second, pair.second);
    }
}

// Test that the trainer starts indexing from the specified starting index
TEST_F(CharTokenizerTrainerTest, TrainWithCustomStartingIndex) {
    std::string text = "abc";
    int starting_index = 10;
    CharTokenizer tokenizer = trainer.train(text, starting_index);

    CharTokenizer::Vocabulary expected_vocabulary = {{'a', 10}, {'b', 11}, {'c', 12}};

    // Verify that the generated vocabulary starts at the correct index
    ASSERT_EQ(tokenizer.get_vocabulary().size(), expected_vocabulary.size());

    for (const auto& pair : expected_vocabulary) {
        auto it = tokenizer.get_vocabulary().find(pair.first);
        ASSERT_NE(it, tokenizer.get_vocabulary().end());
        ASSERT_EQ(it->second, pair.second);
    }
}

// Test that the trainer handles an empty string correctly
TEST_F(CharTokenizerTrainerTest, TrainWithEmptyString) {
    std::string text = "";
    CharTokenizer tokenizer = trainer.train(text);

    // Verify that the generated vocabulary is empty
    ASSERT_TRUE(tokenizer.get_vocabulary().empty());
}
