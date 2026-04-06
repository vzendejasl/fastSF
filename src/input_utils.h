#ifndef FASTSF_INPUT_UTILS_H
#define FASTSF_INPUT_UTILS_H

#include <string>
#include <utility>
#include <vector>

struct TxtChunk {
    long long offset;
    long count;
};

struct SampledTxtData {
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;
    std::vector<double> v1;
    std::vector<double> v2;
    std::vector<double> v3;
};

struct StructuredGridInfo {
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;
    double dx;
    double dy;
    double dz;
};

struct ResolvedInputPath {
    std::string full_path;
    std::string path_without_extension;
    std::string extension;
};

bool file_exists(const std::string& path);
ResolvedInputPath resolve_input_path(const std::string& requested_path, const std::vector<std::string>& search_dirs);
std::pair<std::vector<std::string>, int> detect_txt_header(const std::string& txt_path);
std::vector<TxtChunk> build_txt_chunk_index(const std::string& txt_path, int skip_count, long chunk_size = 1000000);
SampledTxtData read_txt_chunk(const std::string& txt_path, long long offset, long count, bool is_scalar);
double validate_uniform_axis(const std::vector<double>& axis, const std::string& name);
std::vector<int> compute_axis_indices(const std::vector<double>& values, const std::vector<double>& axis, double spacing, const std::string& name);
StructuredGridInfo infer_structured_grid(const SampledTxtData& data);
std::string basename_without_extension(const std::string& path);
std::string dataset_leaf_name(const std::string& dataset_path, const std::string& fallback);

#endif
