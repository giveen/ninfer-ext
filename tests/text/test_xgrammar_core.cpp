// Smoke coverage for the vendored grammar source base. End-to-end constraint behavior belongs to the
// text adapter and the Engine tests; this protects the import itself: that the CPU core builds under
// this toolchain and that the vocabulary-wide matcher contract the sampling side consumes is intact.
#include <dlpack/dlpack.h>
#include <xgrammar/grammar.h>
#include <xgrammar/matcher.h>
#include <xgrammar/tokenizer_info.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

// The vocabulary the sampling side classifies for a model: raw token bytes plus the stop token.
const std::vector<std::string> kVocab{"a", "b", "c", "d", "ab", "abd", "x", "<|end|>"};
constexpr std::int32_t kStop = 7;

// A caller-owned bitmask over the vocabulary, filled by the matcher.
class Mask {
public:
    Mask()
        : words_(static_cast<std::size_t>(
              xgrammar::GetBitmaskSize(static_cast<int>(kVocab.size()))),
          0) {
        shape_        = static_cast<std::int64_t>(words_.size());
        tensor_.data  = words_.data();
        tensor_.device = DLDevice{kDLCPU, 0};
        tensor_.ndim  = 1;
        tensor_.dtype = DLDataType{static_cast<std::uint8_t>(kDLInt), 32, 1};
        tensor_.shape = &shape_;
        tensor_.strides = nullptr;
        tensor_.byte_offset = 0;
    }

    void reset() { std::fill(words_.begin(), words_.end(), 0); }

    DLTensor* tensor() { return &tensor_; }

    // The allowed token ids in order, e.g. "0,4,5": compact enough to assert the whole set.
    std::string ids() const {
        std::string out;
        for (std::size_t id = 0; id < kVocab.size(); ++id) {
            const auto token = static_cast<std::int32_t>(id);
            if (((words_[id / 32U] >> (token % 32)) & 1) == 0) { continue; }
            if (!out.empty()) { out += ','; }
            out += std::to_string(id);
        }
        return out;
    }

private:
    std::vector<std::int32_t> words_;
    std::int64_t shape_ = 0;
    DLTensor tensor_{};
};

std::vector<std::int32_t> stop_ids() { return {kStop}; }

xgrammar::TokenizerInfo tokenizer() {
    return xgrammar::TokenizerInfo(kVocab, xgrammar::VocabType::RAW, std::nullopt, stop_ids());
}

// A fixed sequence with one branch, so every position has a single legal continuation beyond the
// branch itself.
constexpr const char* kSequence = R"(root ::= "a" ("b" | "c") "d")";
constexpr const char* kOpen     = R"(root ::= [a-z]+)";

void test_sequence() {
    xgrammar::GrammarCompiler compiler(tokenizer());
    const auto compiled = compiler.CompileGrammar(xgrammar::Grammar::FromEBNF(kSequence));
    xgrammar::GrammarMatcher matcher(compiled);
    Mask mask;

    // Tokens spanning more than one terminal stay legal, and the stop token is not offered while the
    // root rule cannot complete.
    check(matcher.FillNextTokenBitmask(mask.tensor()),
          "a constrained state reports that its mask has to be applied");
    check(mask.ids() == "0,4,5", "the initial mask allows exactly the completing tokens, got " + mask.ids());

    check(!matcher.AcceptToken(6), "a token outside the language is rejected");
    check(matcher.AcceptToken(0), "the first terminal is accepted");
    mask.reset();
    matcher.FillNextTokenBitmask(mask.tensor());
    check(mask.ids() == "1,2", "the branch decides the legal set, got " + mask.ids());

    check(matcher.AcceptToken(1), "the branch terminal is accepted");
    mask.reset();
    matcher.FillNextTokenBitmask(mask.tensor());
    check(mask.ids() == "3", "the closing terminal is the only continuation, got " + mask.ids());
    check(matcher.AcceptToken(3), "the closing terminal is accepted");

    // A satisfied root rule still requires the stop token: the two facts are distinct.
    check(matcher.IsCompleted() && !matcher.IsTerminated(),
          "a satisfied root rule is not termination");
    mask.reset();
    matcher.FillNextTokenBitmask(mask.tensor());
    check(mask.ids() == "7", "a satisfied root rule leaves the stop token, got " + mask.ids());

    check(matcher.AcceptToken(kStop) && matcher.IsTerminated(), "the stop token terminates the round");
    matcher.Rollback(1);
    check(!matcher.IsTerminated(), "rollback leaves the terminated state");
    mask.reset();
    matcher.FillNextTokenBitmask(mask.tensor());
    check(mask.ids() == "7", "rollback returns to the state before the stop token, got " + mask.ids());
}

void test_multi_terminal_token() {
    xgrammar::GrammarCompiler compiler(tokenizer());
    xgrammar::GrammarMatcher matcher(
        compiler.CompileGrammar(xgrammar::Grammar::FromEBNF(kSequence)));
    check(matcher.AcceptToken(5) && matcher.IsCompleted(),
          "a token spanning every terminal is accepted from the initial state");
}

void test_continuation_prefix() {
    xgrammar::GrammarCompiler compiler(tokenizer());
    xgrammar::GrammarMatcher matcher(
        compiler.CompileGrammar(xgrammar::Grammar::FromEBNF(kSequence)));
    check(matcher.AcceptString("ab"), "a continuation prefix is accepted as one step");
    Mask mask;
    matcher.FillNextTokenBitmask(mask.tensor());
    check(mask.ids() == "3", "the continuation prefix advances the language state, got " + mask.ids());
}

void test_open_language() {
    xgrammar::GrammarCompiler compiler(tokenizer());
    xgrammar::GrammarMatcher matcher(compiler.CompileGrammar(xgrammar::Grammar::FromEBNF(kOpen)));
    check(matcher.AcceptToken(0) && matcher.IsCompleted(),
          "an open language is satisfied after one letter");

    // Satisfaction does not stop generation: every letter keeps its place in the legal set and the
    // stop token joins it. With the whole vocabulary legal, the mask reports that it is unnecessary.
    Mask mask;
    check(!matcher.FillNextTokenBitmask(mask.tensor()),
          "an unconstrained state reports that its mask is unnecessary");
    check(mask.ids() == "0,1,2,3,4,5,6,7",
          "an open language allows every letter and the stop token, got " + mask.ids());
    check(matcher.AcceptToken(kStop) && matcher.IsTerminated(), "the stop token ends the round");
}

void test_fork_isolation() {
    xgrammar::GrammarCompiler compiler(tokenizer());
    xgrammar::GrammarMatcher matcher(
        compiler.CompileGrammar(xgrammar::Grammar::FromEBNF(kSequence)));
    check(matcher.AcceptToken(0), "the first terminal is accepted");
    auto fork = matcher.Fork();
    check(fork.AcceptToken(1), "the fork accepts a branch terminal");

    Mask original;
    matcher.FillNextTokenBitmask(original.tensor());
    check(original.ids() == "1,2", "a fork does not advance the matcher it came from, got " + original.ids());
    Mask forked;
    fork.FillNextTokenBitmask(forked.tensor());
    check(forked.ids() == "3", "the fork continues from its own state, got " + forked.ids());
}

void test_invalid_grammar() {
    bool threw = false;
    std::string message;
    try {
        (void)xgrammar::Grammar::FromEBNF("root ::= (");
    } catch (const std::exception& error) {
        threw   = true;
        message = error.what();
    }
    check(threw && !message.empty(), "an unparsable grammar reports a diagnosable error");
}

}  // namespace

int main() {
    test_sequence();
    test_multi_terminal_token();
    test_continuation_prefix();
    test_open_language();
    test_fork_isolation();
    test_invalid_grammar();
    if (failures != 0) {
        std::cerr << failures << " xgrammar source-base checks failed\n";
        return 1;
    }
    std::cout << "xgrammar source-base checks passed\n";
    return 0;
}
