#include "input_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>

namespace {

double round_coord(double value) {
    return std::round(value * 1.0e10) / 1.0e10;
}

std::string strip_extension(const std::string& path) {
    std::size_t slash = path.find_last_of("/\\");
    std::size_t dot = path.find_last_of('.');
    if (dot == std::string::npos || (slash != std::string::npos && dot < slash + 1)) return path;
    return path.substr(0, dot);
}

std::string extension_of(const std::string& path) {
    std::size_t slash = path.find_last_of("/\\");
    std::size_t dot = path.find_last_of('.');
    if (dot == std::string::npos || (slash != std::string::npos && dot < slash + 1)) return "";
    return path.substr(dot);
}

std::string parent_dir(const std::string& path) {
    std::size_t slash = path.find_last_of("/\\");
    if (slash == std::string::npos) return "";
    return path.substr(0, slash);
}

std::string leaf_name(const std::string& path) {
    std::size_t slash = path.find_last_of("/\\");
    if (slash == std::string::npos) return path;
    return path.substr(slash + 1);
}

std::string join_path(const std::string& dir, const std::string& leaf) {
    if (dir.empty()) return leaf;
    if (leaf.empty()) return dir;
    if (dir[dir.size() - 1] == '/') return dir + leaf;
    return dir + "/" + leaf;
}

std::vector<double> sorted_unique_rounded(const std::vector<double>& values) {
    std::vector<double> rounded;
    rounded.reserve(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) rounded.push_back(round_coord(values[i]));
    std::sort(rounded.begin(), rounded.end());
    rounded.erase(std::unique(rounded.begin(), rounded.end()), rounded.end());
    return rounded;
}

}  // namespace

bool file_exists(const std::string& path) {
    std::ifstream input(path.c_str(), std::ios::binary);
    return input.good();
}

ResolvedInputPath resolve_input_path(const std::string& requested_path, const std::vector<std::string>& search_dirs) {
    ResolvedInputPath resolved;
    std::string requested_ext = extension_of(requested_path);
    std::vector<std::string> candidate_extensions;
    if (requested_ext == ".h5" || requested_ext == ".txt") candidate_extensions.push_back(requested_ext);
    else {
        candidate_extensions.push_back(".h5");
        candidate_extensions.push_back(".txt");
    }

    std::vector<std::string> candidate_dirs;
    if (!parent_dir(requested_path).empty()) candidate_dirs.push_back(parent_dir(requested_path));
    else {
        candidate_dirs.push_back("");
        for (std::size_t i = 0; i < search_dirs.size(); ++i) {
            if (search_dirs[i] != "") candidate_dirs.push_back(search_dirs[i]);
        }
    }

    std::string stem = strip_extension(leaf_name(requested_path));
    for (std::size_t i = 0; i < candidate_dirs.size(); ++i) {
        if (requested_ext == ".h5" || requested_ext == ".txt") {
            std::string exact = join_path(candidate_dirs[i], leaf_name(requested_path));
            if (file_exists(exact)) {
                resolved.full_path = exact;
                resolved.path_without_extension = strip_extension(exact);
                resolved.extension = requested_ext;
                return resolved;
            }
        }

        for (std::size_t j = 0; j < candidate_extensions.size(); ++j) {
            std::string candidate = join_path(candidate_dirs[i], stem + candidate_extensions[j]);
            if (file_exists(candidate)) {
                resolved.full_path = candidate;
                resolved.path_without_extension = strip_extension(candidate);
                resolved.extension = candidate_extensions[j];
                return resolved;
            }
        }
    }

    throw std::runtime_error("Unable to locate input '" + requested_path + "' as .h5 or .txt");
}

std::pair<std::vector<std::string>, int> detect_txt_header(const std::string& txt_path) {
    std::ifstream input(txt_path.c_str());
    if (!input.is_open()) throw std::runtime_error("Unable to open text file '" + txt_path + "'");
    std::vector<std::string> header_lines;
    std::string line;
    while (std::getline(input, line)) {
        std::string stripped = line;
        std::size_t first = stripped.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) {
            header_lines.push_back(line);
            continue;
        }
        stripped = stripped.substr(first);
        std::istringstream parser(stripped);
        double value;
        bool all_numeric = true;
        while (parser >> value) {}
        if (!parser.eof()) all_numeric = false;
        if (all_numeric) break;
        header_lines.push_back(line);
    }
    return std::make_pair(header_lines, static_cast<int>(header_lines.size()));
}

std::vector<TxtChunk> build_txt_chunk_index(const std::string& txt_path, int skip_count, long chunk_size) {
    std::vector<TxtChunk> chunks;
    std::ifstream input(txt_path.c_str(), std::ios::binary);
    if (!input.is_open()) throw std::runtime_error("Unable to open text file '" + txt_path + "'");
    std::string line;
    for (int i = 0; i < skip_count; ++i) {
        if (!std::getline(input, line)) return chunks;
    }
    while (true) {
        std::streampos offset = input.tellg();
        if (offset < 0) break;
        long count = 0;
        for (long i = 0; i < chunk_size; ++i) {
            if (!std::getline(input, line)) break;
            if (!line.empty()) count++;
        }
        if (count == 0) break;
        TxtChunk chunk;
        chunk.offset = static_cast<long long>(offset);
        chunk.count = count;
        chunks.push_back(chunk);
    }
    return chunks;
}

SampledTxtData read_txt_chunk(const std::string& txt_path, long long offset, long count, bool is_scalar) {
    SampledTxtData data;
    std::ifstream input(txt_path.c_str(), std::ios::binary);
    if (!input.is_open()) throw std::runtime_error("Unable to open text file '" + txt_path + "'");
    input.seekg(static_cast<std::streamoff>(offset));
    std::string line;
    for (long i = 0; i < count; ++i) {
        if (!std::getline(input, line)) break;
        if (line.empty()) continue;
        std::istringstream parser(line);
        std::vector<double> cols;
        double value = 0.0;
        while (parser >> value) cols.push_back(value);
        if (cols.empty()) continue;
        if (!parser.eof()) throw std::runtime_error("Malformed sampled-data row in '" + txt_path + "'");
        std::size_t expected_cols = is_scalar ? 4 : 6;
        if (cols.size() < expected_cols) throw std::runtime_error("Insufficient columns in sampled-data row from '" + txt_path + "'");
        data.x.push_back(cols[0]);
        data.y.push_back(cols[1]);
        data.z.push_back(cols[2]);
        data.v1.push_back(cols[3]);
        if (is_scalar) {
            data.v2.push_back(0.0);
            data.v3.push_back(0.0);
        } else {
            data.v2.push_back(cols[4]);
            data.v3.push_back(cols[5]);
        }
    }
    return data;
}

double validate_uniform_axis(const std::vector<double>& axis, const std::string& name) {
    if (axis.size() <= 1) return 1.0;
    double spacing = axis[1] - axis[0];
    double tol = std::max(1.0, std::abs(spacing)) * 1.0e-8;
    for (std::size_t i = 2; i < axis.size(); ++i) {
        double diff = axis[i] - axis[i - 1];
        if (std::abs(diff - spacing) > tol) throw std::runtime_error("Axis '" + name + "' is not uniformly spaced");
    }
    return spacing;
}

std::vector<int> compute_axis_indices(const std::vector<double>& values, const std::vector<double>& axis, double spacing, const std::string& name) {
    std::vector<int> indices(values.size(), 0);
    if (axis.empty()) throw std::runtime_error("Axis '" + name + "' is empty");
    if (axis.size() == 1) return indices;
    double tol = std::max(1.0, std::abs(spacing)) * 1.0e-8;
    for (std::size_t i = 0; i < values.size(); ++i) {
        int idx = static_cast<int>(std::llround((round_coord(values[i]) - axis[0]) / spacing));
        if (idx < 0 || idx >= static_cast<int>(axis.size())) throw std::runtime_error("Axis index out of bounds for '" + name + "'");
        if (std::abs(axis[idx] - round_coord(values[i])) > tol) throw std::runtime_error("Axis '" + name + "' contains coordinates that do not map cleanly to the structured grid");
        indices[i] = idx;
    }
    return indices;
}

StructuredGridInfo infer_structured_grid(const SampledTxtData& data) {
    StructuredGridInfo info;
    info.x = sorted_unique_rounded(data.x);
    info.y = sorted_unique_rounded(data.y);
    info.z = sorted_unique_rounded(data.z);
    info.dx = validate_uniform_axis(info.x, "x");
    info.dy = validate_uniform_axis(info.y, "y");
    info.dz = validate_uniform_axis(info.z, "z");
    return info;
}

std::string basename_without_extension(const std::string& path) {
    return strip_extension(leaf_name(path));
}

std::string dataset_leaf_name(const std::string& dataset_path, const std::string& fallback) {
    std::string leaf = leaf_name(dataset_path);
    return leaf.empty() ? fallback : leaf;
}
