#pragma once

#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <stdexcept>
#include <string>

namespace cudallm {

// Minimal "--key value" / "--flag" parser for the tools. A token is a flag when it is the
// last argument or the next token also starts with "--". Unknown keys print usage and exit.
class Args {
public:
    Args(int argc, char** argv, std::string usage, std::set<std::string> known) : usage_(std::move(usage)) {
        known.insert("help");
        for (int i = 1; i < argc; ++i) {
            std::string tok = argv[i];
            if (tok.rfind("--", 0) != 0) fail("unexpected argument '" + tok + "'");
            std::string key = tok.substr(2);
            if (!known.count(key)) fail("unknown option '--" + key + "'");
            if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                values_[key] = argv[++i];
            } else {
                values_[key] = "";
                flags_.insert(key);
            }
        }
        if (values_.count("help")) {
            std::printf("%s\n", usage_.c_str());
            std::exit(0);
        }
    }

    bool has(const std::string& key) const { return values_.count(key) != 0; }

    std::string get_str(const std::string& key, const std::string& def) const {
        auto it = values_.find(key);
        if (it == values_.end()) return def;
        if (flags_.count(key)) fail("option '--" + key + "' needs a value");
        return it->second;
    }

    int get_int(const std::string& key, int def) const {
        if (!has(key)) return def;
        const std::string v = get_str(key, "");
        try {
            size_t n = 0;
            const int r = std::stoi(v, &n);
            if (n != v.size()) throw std::invalid_argument(v);
            return r;
        } catch (const std::exception&) {
            fail("option '--" + key + "' expects an integer, got '" + v + "'");
        }
        return def;
    }

    float get_float(const std::string& key, float def) const {
        if (!has(key)) return def;
        const std::string v = get_str(key, "");
        try {
            size_t n = 0;
            const float r = std::stof(v, &n);
            if (n != v.size()) throw std::invalid_argument(v);
            return r;
        } catch (const std::exception&) {
            fail("option '--" + key + "' expects a number, got '" + v + "'");
        }
        return def;
    }

    // "--flag" alone is true; "--flag 0|1|true|false" sets it explicitly.
    bool get_bool(const std::string& key, bool def) const {
        auto it = values_.find(key);
        if (it == values_.end()) return def;
        if (flags_.count(key)) return true;
        const std::string& v = it->second;
        if (v == "1" || v == "true") return true;
        if (v == "0" || v == "false") return false;
        fail("option '--" + key + "' expects 0/1/true/false, got '" + v + "'");
        return def;
    }

    [[noreturn]] void fail(const std::string& msg) const {
        std::fprintf(stderr, "error: %s\n\n%s\n", msg.c_str(), usage_.c_str());
        std::exit(2);
    }

private:
    std::string usage_;
    std::map<std::string, std::string> values_;
    std::set<std::string> flags_;
};

}  // namespace cudallm
