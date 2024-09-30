#include "msgpack_file.hpp"

#include <fstream>
#define MSGPACK_NO_BOOST
#include <fstream>
#include <msgpack.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace msgpack {
MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS) {
    namespace adaptor {

    // Custom adaptor for std::variant
    template <typename... Types>
    struct pack<std::variant<Types...>> {
        template <typename Stream>
        packer<Stream>& operator()(msgpack::packer<Stream>& o, const std::variant<Types...>& v) const {
            // Pack the index of the active type and the value
            o.pack_array(2);
            o.pack(v.index());
            std::visit([&o](const auto& val) { o.pack(val); }, v);
            return o;
        }
    };

    template <typename... Types>
    struct convert<std::variant<Types...>> {
        msgpack::object const& operator()(msgpack::object const& o, std::variant<Types...>& v) const {
            if (o.type != msgpack::type::ARRAY || o.via.array.size != 2) {
                throw msgpack::type_error();
            }

            std::size_t index = o.via.array.ptr[0].as<std::size_t>();

            auto& obj = o.via.array.ptr[1];

            // Helper lambda to set the variant based on index
            bool success = set_variant_by_index(index, obj, v);
            if (!success) {
                throw msgpack::type_error();
            }

            return o;
        }

    private:
        template <std::size_t N = 0>
        bool set_variant_by_index(std::size_t index, msgpack::object const& obj, std::variant<Types...>& v) const {
            if constexpr (N < sizeof...(Types)) {
                if (index == N) {
                    using T = std::variant_alternative_t<N, std::variant<Types...>>;
                    T val;
                    obj.convert(val);
                    v = std::move(val);
                    return true;
                } else {
                    return set_variant_by_index<N + 1>(index, obj, v);
                }
            } else {
                return false;
            }
        }
    };

    }  // namespace adaptor
}  // namespace MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
}  // namespace msgpack

namespace ttml::serialization {
class MsgPackFile::Impl {
public:
    Impl() {}
    ~Impl() {}

    // Methods to store different types
    void put(std::string_view key, int value) { m_data[std::string(key)] = value; }

    void put(std::string_view key, float value) { m_data[std::string(key)] = value; }

    void put(std::string_view key, double value) { m_data[std::string(key)] = value; }

    void put(std::string_view key, uint32_t value) { m_data[std::string(key)] = value; }

    void put(std::string_view key, const std::string& value) { m_data[std::string(key)] = value; }

    void put(std::string_view key, std::string_view value) { m_data[std::string(key)] = std::string(value); }

    // Overloads for std::span
    void put(std::string_view key, std::span<const int> value) {
        m_data[std::string(key)] = std::vector<int>(value.begin(), value.end());
    }

    void put(std::string_view key, std::span<const float> value) {
        m_data[std::string(key)] = std::vector<float>(value.begin(), value.end());
    }

    void put(std::string_view key, std::span<const double> value) {
        m_data[std::string(key)] = std::vector<double>(value.begin(), value.end());
    }

    void put(std::string_view key, std::span<const uint32_t> value) {
        m_data[std::string(key)] = std::vector<uint32_t>(value.begin(), value.end());
    }

    void put(std::string_view key, std::span<const std::string> value) {
        m_data[std::string(key)] = std::vector<std::string>(value.begin(), value.end());
    }

    // Serialization method
    void serialize(const std::string& filename) {
        // Create a buffer for packing
        msgpack::sbuffer sbuf;

        // Pack the data into the buffer
        msgpack::pack(sbuf, m_data);

        // Write the buffer to a file
        std::ofstream ofs(filename, std::ios::binary);
        if (ofs.is_open()) {
            ofs.write(sbuf.data(), static_cast<std::streamsize>(sbuf.size()));
            ofs.close();
        } else {
            throw std::runtime_error("Unable to open file for writing: " + filename);
        }
    }

    // Deserialization method
    void deserialize(const std::string& filename) {
        // Read the file content into a string buffer
        std::ifstream ifs(filename, std::ios::binary);
        if (!ifs.is_open()) {
            throw std::runtime_error("Unable to open file for reading: " + filename);
        }
        std::string buffer((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
        ifs.close();

        // Unpack the buffer into msgpack object
        msgpack::object_handle handle = msgpack::unpack(buffer.data(), buffer.size());

        // Convert the msgpack object to the desired type
        msgpack::object obj = handle.get();

        // Clear existing data
        m_data.clear();

        // Convert object to m_data
        obj.convert(m_data);
    }

    // Methods to get values
    bool get(std::string_view key, int& value) const { return get_value(key, value); }

    bool get(std::string_view key, float& value) const { return get_value(key, value); }

    bool get(std::string_view key, double& value) const { return get_value(key, value); }

    bool get(std::string_view key, uint32_t& value) const { return get_value(key, value); }

    bool get(std::string_view key, std::string& value) const { return get_value(key, value); }

    // Methods to get vectors
    bool get(std::string_view key, std::vector<int>& value) const { return get_value(key, value); }

    bool get(std::string_view key, std::vector<float>& value) const { return get_value(key, value); }

    bool get(std::string_view key, std::vector<double>& value) const { return get_value(key, value); }

    bool get(std::string_view key, std::vector<uint32_t>& value) const { return get_value(key, value); }

    bool get(std::string_view key, std::vector<std::string>& value) const { return get_value(key, value); }

private:
    using ValueType = std::variant<
        int,
        float,
        double,
        uint32_t,
        std::string,
        std::vector<int>,
        std::vector<float>,
        std::vector<double>,
        std::vector<uint32_t>,
        std::vector<std::string>>;

    std::unordered_map<std::string, ValueType> m_data;

    // Helper function to get value from m_data
    template <typename T>
    bool get_value(std::string_view key, T& value) const {
        auto it = m_data.find(std::string(key));
        if (it != m_data.end()) {
            if (const auto* pval = std::get_if<T>(&(it->second))) {
                value = *pval;
                return true;
            } else {
                // Type mismatch
                return false;
            }
        } else {
            // Key not found
            return false;
        }
    }
};

MsgPackFile::MsgPackFile() : m_impl(new Impl()) {}

MsgPackFile::~MsgPackFile() = default;

MsgPackFile::MsgPackFile(MsgPackFile&&) noexcept = default;

void MsgPackFile::put(std::string_view key, int value) { m_impl->put(key, value); }

void MsgPackFile::put(std::string_view key, float value) { m_impl->put(key, value); }

void MsgPackFile::put(std::string_view key, double value) { m_impl->put(key, value); }

void MsgPackFile::put(std::string_view key, uint32_t value) { m_impl->put(key, value); }

void MsgPackFile::put(std::string_view key, std::string_view value) { m_impl->put(key, value); }

void MsgPackFile::put(std::string_view key, std::span<const int> value) { m_impl->put(key, value); }

void MsgPackFile::put(std::string_view key, std::span<const float> value) { m_impl->put(key, value); }

void MsgPackFile::put(std::string_view key, std::span<const double> value) { m_impl->put(key, value); }

void MsgPackFile::put(std::string_view key, std::span<const uint32_t> value) { m_impl->put(key, value); }

void MsgPackFile::put(std::string_view key, std::span<const std::string> value) { m_impl->put(key, value); }

void MsgPackFile::serialize(const std::string& filename) { m_impl->serialize(filename); }

void MsgPackFile::deserialize(const std::string& filename) { m_impl->deserialize(filename); }

bool MsgPackFile::get(std::string_view key, int& value) const { return m_impl->get(key, value); }

bool MsgPackFile::get(std::string_view key, float& value) const { return m_impl->get(key, value); }

bool MsgPackFile::get(std::string_view key, double& value) const { return m_impl->get(key, value); }

bool MsgPackFile::get(std::string_view key, uint32_t& value) const { return m_impl->get(key, value); }

bool MsgPackFile::get(std::string_view key, std::string& value) const { return m_impl->get(key, value); }

bool MsgPackFile::get(std::string_view key, std::vector<int>& value) const { return m_impl->get(key, value); }

bool MsgPackFile::get(std::string_view key, std::vector<float>& value) const { return m_impl->get(key, value); }

bool MsgPackFile::get(std::string_view key, std::vector<double>& value) const { return m_impl->get(key, value); }

bool MsgPackFile::get(std::string_view key, std::vector<uint32_t>& value) const { return m_impl->get(key, value); }

bool MsgPackFile::get(std::string_view key, std::vector<std::string>& value) const { return m_impl->get(key, value); }

}  // namespace ttml::serialization
