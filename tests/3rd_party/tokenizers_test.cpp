
#include <gtest/gtest.h>
#include <tokenizers_cpp.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>

using tokenizers::Tokenizer;

namespace {
std::string load_bytes_from_file(const std::string& path) {
    std::ifstream file_stream(path, std::ios::in | std::ios::binary);
    EXPECT_TRUE(file_stream.is_open());
    std::string data;
    file_stream.seekg(0, std::ios::end);
    auto size = file_stream.tellg();
    file_stream.seekg(0, std::ios::beg);
    data.resize(size);
    file_stream.read(data.data(), size);
    return data;
}

void print_encode_result(const std::vector<int>& ids) {
    std::cout << "tokens=[";
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i != 0)
            std::cout << ", ";
        std::cout << ids[i];
    }
    std::cout << "]" << std::endl;
}

void test_tokenizer(std::unique_ptr<Tokenizer> tok, bool check_id_back = true) {
    // Check #1. Encode and Decode
    std::string prompt = "What is the  capital of Canada?";
    std::vector<int> ids = tok->Encode(prompt);
    std::string decoded_prompt = tok->Decode(ids);
    print_encode_result(ids);
    std::cout << "decode=\"" << decoded_prompt << "\"" << std::endl;
    EXPECT_EQ(decoded_prompt, prompt);

    // Check #2. IdToToken and TokenToId
    std::vector<int32_t> ids_to_test = {0, 1, 2, 3, 32, 33, 34, 130, 131, 1000};
    for (auto id : ids_to_test) {
        auto token = tok->IdToToken(id);
        auto id_new = tok->TokenToId(token);
        std::cout << "id=" << id << ", token=\"" << token << "\", id_new=" << id_new << std::endl;
        if (check_id_back) {
            EXPECT_EQ(id, id_new);
        }
    }

    // Check #3. GetVocabSize
    auto vocab_size = tok->GetVocabSize();
    std::cout << "vocab_size=" << vocab_size << std::endl;

    std::cout << std::endl;
}

}  // namespace

TEST(HuggingFaceTokenizer, Example) {
    std::cout << "Tokenizer: Huggingface" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    // Read blob from file.
    auto blob = load_bytes_from_file(std::string(TEST_DATA_DIR) + "/tokenizer.json");
    // Note: all the current factory APIs takes in-memory blob as input.
    // This gives some flexibility on how these blobs can be read.
    auto tok = Tokenizer::FromBlobJSON(blob);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "Load time: " << duration << " ms" << std::endl;

    test_tokenizer(std::move(tok), true);
}