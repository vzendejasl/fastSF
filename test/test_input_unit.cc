#include "input_utils.h"

#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void write_file(const std::string& path, const std::string& contents) {
    std::ofstream out(path.c_str());
    out << contents;
}

void expect_throw_nonuniform_axis() {
    bool threw = false;
    try {
        std::vector<double> axis;
        axis.push_back(0.0);
        axis.push_back(0.5);
        axis.push_back(1.1);
        validate_uniform_axis(axis, "x");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);
}

}  // namespace

int main() {
    const std::string header_file = "unit_header_detection.txt";
    write_file(header_file, "# comment\n\n# metadata\n0 0 0 1 2 3\n");
    std::pair<std::vector<std::string>, int> header = detect_txt_header(header_file);
    assert(header.second == 3);
    std::remove(header_file.c_str());

    SampledTxtData sampled;
    sampled.x.push_back(1.0); sampled.y.push_back(0.0); sampled.z.push_back(0.0);
    sampled.x.push_back(0.0); sampled.y.push_back(1.0); sampled.z.push_back(0.0);
    sampled.x.push_back(0.0); sampled.y.push_back(0.0); sampled.z.push_back(1.0);
    sampled.x.push_back(1.0); sampled.y.push_back(1.0); sampled.z.push_back(1.0);
    StructuredGridInfo grid = infer_structured_grid(sampled);
    assert(grid.x.size() == 2 && grid.y.size() == 2 && grid.z.size() == 2);
    assert(grid.dx == 1.0 && grid.dy == 1.0 && grid.dz == 1.0);

    std::vector<double> xvals;
    xvals.push_back(0.0);
    xvals.push_back(0.99999999999);
    xvals.push_back(1.00000000001);
    std::vector<double> axis;
    axis.push_back(0.0);
    axis.push_back(1.0);
    std::vector<int> indices = compute_axis_indices(xvals, axis, 1.0, "x");
    assert(indices[0] == 0 && indices[1] == 1 && indices[2] == 1);

    expect_throw_nonuniform_axis();

    const std::string txt_file = "unit_resolve.txt";
    const std::string h5_file = "unit_resolve.h5";
    write_file(txt_file, "0 0 0 1\n");
    write_file(h5_file, "placeholder");
    std::vector<std::string> search_dirs;
    search_dirs.push_back("");
    ResolvedInputPath resolved = resolve_input_path("unit_resolve", search_dirs);
    assert(resolved.extension == ".h5");
    assert(resolved.path_without_extension == "unit_resolve");
    std::remove(txt_file.c_str());
    std::remove(h5_file.c_str());

    std::cout << "input_utils unit tests passed\n";
    return 0;
}
