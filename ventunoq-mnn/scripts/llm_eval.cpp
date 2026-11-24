//
//  llm_eval.cpp — batch eval driver for the ventunoq-mnn quality harness (F14).
//
//  llm_demo's batch mode is one prompt per line and streams all answers into one
//  stdout with no separators; the quality suites need multi-line prompts and clean
//  per-item answers. This tool: one prompt file in, one answer file out, greedy
//  sampling, fresh KV per item, resume-safe (existing outputs are skipped).
//
//  Usage: llm_eval <config.json> <batch_dir> [--max-tokens N] [--base] [--nothink]
//                 [--system TEXT]
//  Reads  batch_dir/prompts/NNNNN.txt
//  Writes batch_dir/outputs/NNNNN.txt   (progress on stderr)
//

#include <llm/llm.hpp>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <dirent.h>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#if !defined(_WIN32) && !defined(_MSC_VER)
#include <limits.h>
#endif

using namespace MNN::Transformer;

// Same convention as llm_demo: the weight-mmap tmp cache carries no model identity,
// so key it by the resolved config path to avoid cross-model cache reuse.
static std::string tmpDirFor(const std::string& configPath) {
    char buf[PATH_MAX];
    std::string key = (nullptr != realpath(configPath.c_str(), buf)) ? buf : configPath;
    std::ostringstream os;
    os << "tmp_eval_" << std::hex << std::hash<std::string>()(key);
    return os.str();
}

static std::string jsonEscape(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if (c == '\n') o += "\\n";
        else if (c == '\t') o += "\\t";
        else o += c;
    }
    return o;
}

static bool readFile(const std::string& path, std::string& text) {
    std::ifstream is(path, std::ios::binary);
    if (!is) return false;
    std::ostringstream os;
    os << is.rdbuf();
    text = os.str();
    return true;
}

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        printf("Usage: %s config.json batch_dir [--max-tokens N] [--base] [--nothink] [--system TEXT]\n", argv[0]);
        return 1;
    }
    std::string configPath = argv[1], batch = argv[2], system;
    int maxTokens = 512;
    bool base = false, nothink = false;
    for (int i = 3; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--max-tokens" && i + 1 < argc) maxTokens = atoi(argv[++i]);
        else if (a == "--base") base = true;
        else if (a == "--nothink") nothink = true;
        else if (a == "--system" && i + 1 < argc) system = argv[++i];
    }

    std::vector<std::string> names;
    {
        DIR* d = opendir((batch + "/prompts").c_str());
        if (!d) { fprintf(stderr, "no batch dir: %s/prompts\n", batch.c_str()); return 1; }
        struct dirent* e;
        while (nullptr != (e = readdir(d))) {
            std::string n = e->d_name;
            if (n.size() > 4 && n.compare(n.size() - 4, 4, ".txt") == 0) names.push_back(n);
        }
        closedir(d);
    }
    std::sort(names.begin(), names.end());
    mkdir((batch + "/outputs").c_str(), 0755);

    std::unique_ptr<Llm> llm(Llm::createLLM(configPath));
    // greedy = the whole point: quality diffs must be engine effects, not sampling
    // noise (same rule as the sibling llama.cpp harness running temp 0).
    std::string cfg = "{\"sampler_type\":\"greedy\",\"tmp_path\":\"" + tmpDirFor(configPath) + "\"";
    if (base) cfg += ",\"use_template\":false";
    if (!system.empty()) cfg += ",\"system_prompt\":\"" + jsonEscape(system) + "\"";
    cfg += "}";
    llm->set_config(cfg);
    if (nothink) llm->set_config(R"({"jinja":{"context":{"enable_thinking":false}}})");
    if (!llm->load()) {
        fprintf(stderr, "LLM init error (F9: is this dir text-only? scripts/text_only.py)\n");
        return 2;
    }
    auto context = llm->getContext();

    int done = 0, skipped = 0;
    for (size_t i = 0; i < names.size(); i++) {
        std::string outPath = batch + "/outputs/" + names[i];
        if (access(outPath.c_str(), F_OK) == 0) { skipped++; continue; }
        std::string prompt;
        if (!readFile(batch + "/prompts/" + names[i], prompt)) {
            fprintf(stderr, "cannot read %s\n", names[i].c_str());
            return 1;
        }
        auto t0 = time(nullptr);
        llm->reset();
        std::ostringstream os;
        llm->response(prompt, &os, nullptr, maxTokens);
        if (context->status == LlmStatus::INTERNAL_ERROR) {
            fprintf(stderr, "INTERNAL_ERROR generating %s\n", names[i].c_str());
            return 3;
        }
        std::ofstream out(outPath, std::ios::binary);
        out << os.str();
        out.close();
        done++;
        fprintf(stderr, "[%zu/%zu] %s chars=%zu gen=%ld tok %.1fs\n", i + 1, names.size(),
                names[i].c_str(), os.str().size(), (long)context->gen_seq_len,
                difftime(time(nullptr), t0));
    }
    printf("DONE items=%zu generated=%d skipped=%d\n", names.size(), done, skipped);
    return 0;
}
