#include "models/qwen3_5/frontend/digest.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace fi = ninfer::models::qwen3_5::frontend;

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

// Deterministic byte pattern for the length-boundary vectors.
std::vector<std::uint8_t> pattern_bytes(std::size_t size) {
    std::vector<std::uint8_t> bytes(size);
    for (std::size_t index = 0; index < size; ++index) {
        bytes[index] = static_cast<std::uint8_t>(index & 0xffU);
    }
    return bytes;
}

int check_digest(std::size_t size, std::string_view expected) {
    const std::string digest = fi::sha256_hex(fi::sha256(pattern_bytes(size)));
    return check(digest == std::string(expected), "sha256 of " + std::to_string(size) +
                                                      " pattern bytes was " + digest +
                                                      ", expected " + std::string(expected));
}

int test_known_answer_vectors() {
    int failures = 0;
    failures += check(fi::sha256_hex(fi::sha256(std::string_view{})) ==
                          "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                      "the empty-message digest changed");
    failures += check(fi::sha256_hex(fi::sha256(std::string_view("abc"))) ==
                          "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
                      "the 'abc' digest changed");
    // The 56-byte vector still fits one padded block; the 112-byte one needs the two-block tail.
    failures += check(fi::sha256_hex(fi::sha256(std::string_view(
                          "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"))) ==
                          "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
                      "the 56-byte vector digest changed");
    failures += check(
        fi::sha256_hex(fi::sha256(std::string_view(
            "abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnop"
            "qrlmnopqrsmnopqrstnopqrstu"))) ==
            "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1",
        "the 112-byte vector digest changed");
    return failures;
}

int test_padding_boundaries() {
    struct Vector {
        std::size_t size;
        std::string_view digest;
    };

    // 55/56 and 63/64 straddle the single-versus-double padded-block decision.
    const Vector vectors[] = {
        {0, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
        {1, "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d"},
        {55, "463eb28e72f82e0a96c0a4cc53690c571281131f672aa229e0d45ae59b598b59"},
        {56, "da2ae4d6b36748f2a318f23e7ab1dfdf45acdc9d049bd80e59de82a60895f562"},
        {63, "29af2686fd53374a36b0846694cc342177e428d1647515f078784d69cdb9e488"},
        {64, "fdeab9acf3710362bd2658cdc9a29e8f9c757fcf9811603a8c447cd1d9151108"},
        {65, "4bfd2c8b6f1eec7a2afeb48b934ee4b2694182027e6d0fc075074f2fabb31781"},
        {119, "da18797ed7c3a777f0847f429724a2d8cd5138e6ed2895c3fa1a6d39d18f7ec6"},
        {120, "f52b23db1fbb6ded89ef42a23ce0c8922c45f25c50b568a93bf1c075420bbb7c"},
        {127, "92ca0fa6651ee2f97b884b7246a562fa71250fedefe5ebf270d31c546bfea976"},
        {128, "471fb943aa23c511f6f72f8d1652d9c880cfa392ad80503120547703e56a2be5"},
        {1023, "84d353b0e5a01f3e12370cd86b340c1fa07fd8a91490d67eb06b91206df54bc4"},
        {1024, "785b0751fc2c53dc14a4ce3d800e69ef9ce1009eb327ccf458afe09c242c26c9"},
    };
    int failures = 0;
    for (const Vector& vector : vectors) { failures += check_digest(vector.size, vector.digest); }
    return failures;
}

int test_checkpoint_contract() {
    const auto count_calls = [](std::size_t size) {
        int calls = 0;
        (void)fi::sha256(pattern_bytes(size), [&calls] { ++calls; });
        return calls;
    };

    int failures = 0;
    failures += check(count_calls(0) == 0 && count_calls(63) == 0,
                      "checkpoint fired for an input with no complete block");
    failures += check(count_calls(64) == 1, "checkpoint did not fire before the first block");
    failures += check(count_calls((1ULL << 20) - 64) == 1,
                      "checkpoint fired more than once below one MiB of complete blocks");
    failures +=
        check(count_calls(1ULL << 20) == 1, "checkpoint fired per block instead of once per MiB");
    failures += check(count_calls((1ULL << 20) + 64) == 2,
                      "checkpoint did not fire again in the second MiB");

    bool propagated = false;
    try {
        (void)fi::sha256(pattern_bytes(64), [] { throw std::runtime_error("cancelled"); });
    } catch (const std::runtime_error&) { propagated = true; }
    failures += check(propagated, "a throwing checkpoint did not propagate to the caller");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_known_answer_vectors();
    failures += test_padding_boundaries();
    failures += test_checkpoint_contract();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
