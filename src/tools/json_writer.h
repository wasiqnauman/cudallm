#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cudallm {

// Minimal streaming JSON writer: tracks whether a comma is needed at each nesting level.
class JsonWriter {
public:
    JsonWriter& begin_object() { open('{'); return *this; }
    JsonWriter& end_object() { close('}'); return *this; }
    JsonWriter& begin_array() { open('['); return *this; }
    JsonWriter& end_array() { close(']'); return *this; }

    JsonWriter& key(const std::string& k) {
        separator();
        write_string(k);
        out_ += ':';
        after_key_ = true;
        return *this;
    }

    JsonWriter& value(const std::string& v) { separator(); write_string(v); return *this; }
    JsonWriter& value(const char* v) { return value(std::string(v)); }
    JsonWriter& value(int v) { separator(); out_ += std::to_string(v); return *this; }
    JsonWriter& value(uint64_t v) { separator(); out_ += std::to_string(v); return *this; }
    JsonWriter& value(bool v) { separator(); out_ += v ? "true" : "false"; return *this; }
    JsonWriter& value(double v) {
        separator();
        if (!std::isfinite(v)) {
            out_ += "null";
        } else {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%.17g", v);
            out_ += buf;
        }
        return *this;
    }

    const std::string& str() const { return out_; }

    void write_file(const std::string& path) const {
        std::ofstream f(path);
        if (!f) throw std::runtime_error("cannot write " + path);
        f << out_ << '\n';
        if (!f) throw std::runtime_error("write failed: " + path);
    }

private:
    void separator() {
        if (after_key_) {
            after_key_ = false;
            return;
        }
        if (!need_comma_.empty()) {
            if (need_comma_.back()) out_ += ',';
            need_comma_.back() = true;
        }
    }
    void open(char c) {
        separator();
        out_ += c;
        need_comma_.push_back(false);
    }
    void close(char c) {
        if (need_comma_.empty()) throw std::logic_error("JsonWriter: unbalanced close");
        need_comma_.pop_back();
        out_ += c;
    }
    void write_string(const std::string& s) {
        out_ += '"';
        for (unsigned char ch : s) {
            switch (ch) {
                case '"': out_ += "\\\""; break;
                case '\\': out_ += "\\\\"; break;
                case '\n': out_ += "\\n"; break;
                case '\r': out_ += "\\r"; break;
                case '\t': out_ += "\\t"; break;
                case '\b': out_ += "\\b"; break;
                case '\f': out_ += "\\f"; break;
                default:
                    if (ch < 0x20) {
                        char buf[8];
                        std::snprintf(buf, sizeof(buf), "\\u%04x", ch);
                        out_ += buf;
                    } else {
                        out_ += static_cast<char>(ch);
                    }
            }
        }
        out_ += '"';
    }

    std::string out_;
    std::vector<bool> need_comma_;
    bool after_key_ = false;
};

}  // namespace cudallm
