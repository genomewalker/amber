// AMBER - Logger utility
// Clean logging with verbosity control and trace file support

#pragma once

#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace amber {

enum class Verbosity { Quiet, Normal, Verbose };

class Logger {
public:
    Verbosity console_level = Verbosity::Normal;

private:
    std::chrono::steady_clock::time_point start_;
    std::ofstream trace_file_;
    std::mutex mutex_;
    bool is_tty_;
    int last_progress_len_ = 0;
    std::string module_name_;
    std::string version_;

    double elapsed() const {
        return std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start_).count();
    }

    std::string timestamp() const {
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        char buf[32];
        std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&time));
        return buf;
    }

    void write_trace(const std::string& line) {
        if (trace_file_.is_open()) {
            trace_file_ << line << "\n";
            trace_file_.flush();
        }
    }

public:
    Logger() : start_(std::chrono::steady_clock::now()),
               is_tty_(isatty(fileno(stderr))) {}

    explicit Logger(const std::string& module_name, const std::string& version = "")
        : start_(std::chrono::steady_clock::now()),
          is_tty_(isatty(fileno(stderr))),
          module_name_(module_name),
          version_(version) {}

    ~Logger() {
        if (trace_file_.is_open()) {
            trace_file_ << "\n[" << timestamp() << "] Run completed in "
                       << std::fixed << std::setprecision(1) << elapsed() << "s\n";
            trace_file_.close();
        }
    }

    void open_trace(const std::string& path) {
        std::lock_guard<std::mutex> lock(mutex_);
        trace_file_.open(path);
        if (trace_file_.is_open()) {
            trace_file_ << "AMBER";
            if (!module_name_.empty()) trace_file_ << " " << module_name_;
            if (!version_.empty()) trace_file_ << " v" << version_;
            trace_file_ << "\n";
            trace_file_ << "Started: " << timestamp() << "\n";
            trace_file_ << std::string(60, '=') << "\n\n";
        }
    }

    void info(const std::string& msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        clear_progress_unlocked();
        if (console_level >= Verbosity::Normal) {
            std::cerr << msg << "\n";
        }
        write_trace("[" + std::to_string(int(elapsed())) + "s] " + msg);
    }

    void detail(const std::string& msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (console_level >= Verbosity::Verbose) {
            clear_progress_unlocked();
            std::cerr << "  " << msg << "\n";
        }
        write_trace("  " + msg);
    }

    void trace(const std::string& msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        write_trace(msg);
    }

    void warn(const std::string& msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        clear_progress_unlocked();
        std::cerr << "Warning: " << msg << "\n";
        write_trace("[WARN] " + msg);
    }

    void error(const std::string& msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        clear_progress_unlocked();
        std::cerr << "Error: " << msg << "\n";
        write_trace("[ERROR] " + msg);
    }

    void progress(const std::string& stage, int current, int total) {
        if (console_level < Verbosity::Normal) return;
        std::lock_guard<std::mutex> lock(mutex_);

        int pct = total > 0 ? (100 * current / total) : 0;
        std::ostringstream ss;
        ss << stage << " " << current << "/" << total << " (" << pct << "%)";
        std::string line = ss.str();

        if (is_tty_) {
            std::cerr << "\r" << line;
            if ((int)line.size() < last_progress_len_) {
                std::cerr << std::string(last_progress_len_ - line.size(), ' ');
            }
            last_progress_len_ = line.size();
            if (current == total) {
                std::cerr << "\n";
                last_progress_len_ = 0;
            }
            std::cerr.flush();
        } else if (current == total || current == 1 || pct % 25 == 0) {
            std::cerr << line << "\n";
        }
    }

    void clear_progress() {
        std::lock_guard<std::mutex> lock(mutex_);
        clear_progress_unlocked();
    }

    void section(const std::string& title) {
        std::lock_guard<std::mutex> lock(mutex_);
        write_trace("\n" + std::string(60, '='));
        write_trace(" " + title);
        write_trace(std::string(60, '='));
    }

    void subsection(const std::string& title) {
        std::lock_guard<std::mutex> lock(mutex_);
        write_trace("\n" + std::string(40, '-'));
        write_trace(" " + title);
        write_trace(std::string(40, '-'));
    }

    void metric(const std::string& name, double value, int precision = 4) {
        std::ostringstream ss;
        ss << "  " << name << ": " << std::fixed << std::setprecision(precision) << value;
        std::lock_guard<std::mutex> lock(mutex_);
        write_trace(ss.str());
    }

    void metric(const std::string& name, int value) {
        std::lock_guard<std::mutex> lock(mutex_);
        write_trace("  " + name + ": " + std::to_string(value));
    }

    void metric(const std::string& name, const std::string& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        write_trace("  " + name + ": " + value);
    }

    void decision(const std::string& type, const std::string& outcome,
                  const std::string& rationale = "") {
        std::lock_guard<std::mutex> lock(mutex_);
        write_trace("[DECISION:" + type + "] " + outcome);
        if (!rationale.empty()) {
            write_trace("  rationale: " + rationale);
        }
    }

    // Log table header for structured output
    void table_header(const std::string& title, const std::vector<std::string>& columns) {
        std::lock_guard<std::mutex> lock(mutex_);
        write_trace("\n[TABLE:" + title + "]");
        std::ostringstream ss;
        for (size_t i = 0; i < columns.size(); i++) {
            if (i > 0) ss << "\t";
            ss << columns[i];
        }
        write_trace(ss.str());
    }

    // Log table row
    void table_row(const std::vector<std::string>& values) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::ostringstream ss;
        for (size_t i = 0; i < values.size(); i++) {
            if (i > 0) ss << "\t";
            ss << values[i];
        }
        write_trace(ss.str());
    }

private:
    void clear_progress_unlocked() {
        if (is_tty_ && last_progress_len_ > 0) {
            std::cerr << "\r" << std::string(last_progress_len_, ' ') << "\r";
            last_progress_len_ = 0;
        }
    }
};

}  // namespace amber
