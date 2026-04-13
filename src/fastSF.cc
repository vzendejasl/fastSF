/********************************************************************************************************************************************
 * fastSF
 *
 * Copyright (C) 2020, Mahendra K. Verma
 *
 * All rights reserved.
 ********************************************************************************************************************************************
 */

#include "h5si.h"
#include "input_utils.h"
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <fstream>
#include <hdf5.h>
#include <sstream>
#include <blitz/array.h>
#include <mpi.h>
#include <sys/time.h>
#include <limits.h>
#include <unistd.h>
#include <vector>
#include <set>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cctype>
#include <cerrno>
#include <stdexcept>
#include <sys/stat.h>

using namespace std;
using namespace blitz;

//Function declarations
void get_Inputs(int argc, char* argv[]); 
void ComputeAndPrintTKE();
void DumpFieldsToTxt();
void write_3D(Array<double,3>, string);
void write_4D(Array<double,4>, string);
void read_2D(Array<double,2>, string, string, string);
void read_3D(Array<double,3>, string, string, string);
string int_to_str(int);
bool str_to_bool(string);
void VECTOR_TEST_CASE_3D();
void VECTOR_TEST_CASE_2D();
void SCALAR_TEST_CASE_2D();
void SCALAR_TEST_CASE_3D();
void compute_time_elapsed(timeval, timeval, double&);
void finalize_mpi_runtime();
void SFunc2D(const Array<double,2>&, const Array<double,2>&);
void SFunc_long_2D(const Array<double,2>&, const Array<double,2>&);
void SFunc3D(const Array<double,3>&, const Array<double,3>&, const Array<double,3>&);
void SFunc_long_3D(const Array<double,3>&, const Array<double,3>&, const Array<double,3>&);
void Read_Init(Array<double,2>&, Array<double,2>&);
void Read_Init(Array<double,3>&, Array<double,3>&, Array<double,3>&);
void Read_Init(Array<double,2>&);
void Read_Init(Array<double,3>&);
void SF_scalar_3D(const Array<double,3>&);
void SF_scalar_2D(const Array<double,2>&);
void Read_fields();
void resize_SFs();
void calc_SFs();
void write_SFs();
void test_cases();
void show_checklist();
bool compare(Array<int,1> , Array<int,1> );
void calculate_grid_spacing();
void resize_input();
void get_input_shape(string , string , string , Array<int,1>&);
void help_command();
void get_rank(int rank, int py, int& rankx, int& ranky);
void compute_index_list(Array<int,1>& index_list, int Nx, int px, int rank);
void compute_index_list(Array<int,3>& index_list, int Nx, int Ny);
void abort_with_error(const string& message);
void abort_parallel_conversion(const string& message);
void prepare_input_sources();
void prepare_scalar_input();
void prepare_velocity_inputs();
void convert_velocity_txt_to_structured_h5(const string& txt_path, const string& h5_path);
void convert_scalar_txt_to_structured_h5(const string& txt_path, const string& h5_path, const string& field_path);
bool load_structured_grid_spacing(const string& h5_path);
bool h5_path_exists(hid_t object_id, const string& path);
bool is_structured_velocity_h5(const string& h5_path);
bool is_structured_scalar_h5(const string& h5_path, string& field_path);
void write_string_attribute(hid_t object_id, const string& name, const string& value);
void write_int_attribute(hid_t object_id, const string& name, int value);
void write_double_attribute(hid_t object_id, const string& name, double value);
void write_double_dataset_1d(hid_t file_id, const string& dataset_path, const std::vector<double>& values);
void write_double_dataset_3d(hid_t file_id, const string& dataset_path, const std::vector<double>& values, hsize_t nx, hsize_t ny, hsize_t nz);
std::vector<double> read_double_dataset_1d(hid_t file_id, const string& dataset_path);
string default_scalar_field_path();
std::vector<std::pair<int,int> > split_axis_ranges(int length, int parts);
void broadcast_txt_chunks(std::vector<TxtChunk>& chunks, int header_lines);
std::vector<double> gather_unique_axis_values(const std::set<double>& local_values);
int owner_rank_for_x_index(int ix, const std::vector<std::pair<int,int> >& x_ranges);
void write_parallel_velocity_h5(const string& h5_path, const StructuredGridInfo& grid, int x_start, int x_stop, const std::vector<double>& local_vx, const std::vector<double>& local_vy, const std::vector<double>& local_vz);
void write_parallel_scalar_h5(const string& h5_path, const string& field_path, const StructuredGridInfo& grid, int x_start, int x_stop, const std::vector<double>& local_field);
bool axis_has_periodic_duplicate(const std::vector<double>& axis, double domain_length);
std::vector<double> trim_periodic_axis(const std::vector<double>& axis, double domain_length);
bool read_bool_attribute(hid_t object_id, const std::string& name, bool default_value);
void print_stage(const std::string& message);
struct ProgressState {
    int next_percent;
    int next_offset_report;
    int offset_report_stride;
    double start_time;
    double last_log_time;
    double last_detail_time;
    double heartbeat_seconds;
};
ProgressState make_progress_state(double heartbeat_seconds = 120.0, int first_percent = 1);
void print_progress_start(const std::string& label, int total, int x, int y, int z);
void print_loop_progress(const std::string& label, int completed, int total, ProgressState& state, int x, int y, int z);
void print_offset_stage(const std::string& label, const std::string& stage, ProgressState& state, int x, int y, int z, bool force);
void print_q_stage(const std::string& label, const std::string& stage, ProgressState& state, int x, int y, int z, int q_order, bool force);
void print_run_summary();
void configure_output_location(const std::string& input_path);
void ensure_directory_exists(const std::string& dir_path);
std::string output_file_path(const std::string& file_stem);

// Global variables
Array <double,3> T, V1, V2, V3;
Array <double,2> T_2D, V1_2D, V3_2D;
Array<double,4> SF_Grid_pll, SF_Grid_perp, SF_Grid_scalar;
Array<double,3> SF_Grid2D_pll, SF_Grid2D_perp, SF_Grid2D_scalar;
bool two_dimension_switch, scalar_switch, test_switch, longitudinal, dump_switch=false;
int Nx, Ny, Nz, q1, q2, rank_mpi, P, px;
double dx, dy, dz, Lx, Ly, Lz;
string UName="U.V1r", VName="U.V2r", WName="U.V3r", TName="T.Fr";
string UdName="U.V1r", VdName="U.V2r", WdName="U.V3r", TdName="T.Fr";
string SF_Grid_pll_name = "SF_Grid_pll", SF_Grid_perp_name = "SF_Grid_perp", SF_Grid_scalar_name = "SF_Grid_scalar";
string output_dir = "out";

int main(int argc, char *argv[]) {
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_mpi);
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    h5::init();
    timeval start_pt, end_pt, start_t, end_t;
    gettimeofday(&start_t,NULL);
    get_Inputs(argc, argv);
    Read_fields();
    if (dump_switch) {
        DumpFieldsToTxt();
        if (rank_mpi == 0) cout << "Fields dumped to 'in/'. Exiting." << endl;
        h5::finalize(); finalize_mpi_runtime(); exit(0);
    }
    if (rank_mpi==0) {
    	cout<<"\nNumber of processors in x direction: "<<px<<endl;
    	if (two_dimension_switch) cout<<"Number of processors in z direction: "<<P/px<<endl;
    	else cout<<"Number of processors in y direction: "<<P/px<<endl;
  	}  
    print_run_summary();
 	if (px > P || (Nx > 1 && (Nx/2)%px != 0)) {
        if (rank_mpi==0) cout<<"ERROR in processor configuration! Aborting.."<<endl;
        h5::finalize(); finalize_mpi_runtime(); exit(1);
    }
    print_stage("Allocating structure-function output arrays...");
    resize_SFs();
    print_stage("Starting structure-function accumulation...");
    gettimeofday(&start_pt,NULL);
    calc_SFs();
    gettimeofday(&end_pt,NULL);
    print_stage("Structure-function accumulation finished.");
    print_stage("Writing output HDF5 files...");
    write_SFs();
    if (test_switch) test_cases();
    gettimeofday(&end_t,NULL);
    double elapsedt, elapsepdt;
    compute_time_elapsed(start_t, end_t, elapsedt);
    compute_time_elapsed(start_pt, end_pt, elapsepdt);
    if (rank_mpi==0) {
        cout<<"\nTime elapsed for the parallel part: "<<elapsepdt<<endl;
        cout<<"\nTotal time elapsed: "<<elapsedt<<endl;
        cout<<"\nProgram ends."<<endl;
   }
    h5::finalize();
    finalize_mpi_runtime();
    return 0;
}

void finalize_mpi_runtime() {
    const char* skip = std::getenv("FASTSF_SKIP_MPI_FINALIZE");
    if (!(skip && std::string(skip) == "1")) MPI_Finalize();
}

void calculate_grid_spacing(){
    dx = (Nx<=1) ? 0 : Lx/double(Nx-1);
    dy = (Ny<=1) ? 0 : Ly/double(Ny-1);
    dz = (Nz<=1) ? 0 : Lz/double(Nz-1);
}

void abort_with_error(const string& message) {
    if (rank_mpi == 0) cerr << message << endl;
    h5::finalize();
    finalize_mpi_runtime();
    exit(1);
}

void abort_parallel_conversion(const string& message) {
    if (rank_mpi == 0) cerr << message << endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
}

void print_stage(const std::string& message) {
    if (rank_mpi == 0) cout << "[fastSF] " << message << endl;
}

ProgressState make_progress_state(double heartbeat_seconds, int first_percent) {
    ProgressState state;
    state.next_percent = first_percent;
    state.next_offset_report = 100;
    state.offset_report_stride = 100;
    state.start_time = MPI_Wtime();
    state.last_log_time = state.start_time;
    state.last_detail_time = state.start_time;
    state.heartbeat_seconds = heartbeat_seconds;
    return state;
}

void print_progress_start(const std::string& label, int total, int x, int y, int z) {
    if (rank_mpi != 0 || total <= 0) return;
    cout << "[fastSF] " << label << ": 0% (0/" << total << ")";
    if (y >= 0) cout << ", starting lag idx=(" << x << "," << y << "," << z << ")";
    else cout << ", starting lag idx=(" << x << "," << z << ")";
    cout << endl;
}

void print_loop_progress(const std::string& label, int completed, int total, ProgressState& state, int x, int y, int z) {
    if (rank_mpi != 0 || total <= 0) return;
    double now = MPI_Wtime();
    int pct = static_cast<int>((100.0 * completed) / total);
    double elapsed = now - state.start_time;
    bool finished = (completed >= total);
    bool crossed_percent = (pct >= state.next_percent);
    bool crossed_offset_report = (completed >= state.next_offset_report);
    bool heartbeat = (!finished && (now - state.last_log_time) >= state.heartbeat_seconds);

    if (!finished && !crossed_percent && !crossed_offset_report && !heartbeat) return;

    int display_pct = finished ? 100 : pct;
    if (crossed_percent && !finished) state.next_percent = pct + 1;
    if (crossed_offset_report && !finished) {
        while (completed >= state.next_offset_report) state.next_offset_report += state.offset_report_stride;
    }
    if (finished) state.next_percent = 101;

    cout << "[fastSF] " << label << ": " << display_pct << "% (" << completed << "/" << total << ")";
    if (completed > 0 && elapsed > 0.0) {
        double rate = completed / elapsed;
        double remaining = (rate > 0.0) ? (total - completed) / rate : 0.0;
        cout << ", elapsed " << elapsed << " s";
        if (!finished) cout << ", ETA " << remaining << " s";
    }
    if (y >= 0) cout << ", current lag idx=(" << x << "," << y << "," << z << ")";
    else cout << ", current lag idx=(" << x << "," << z << ")";
    cout << endl;
    state.last_log_time = now;
    state.last_detail_time = now;
}

void print_offset_stage(const std::string& label, const std::string& stage, ProgressState& state, int x, int y, int z, bool force) {
    if (rank_mpi != 0) return;
    double now = MPI_Wtime();
    if (!force && (now - state.last_detail_time) < state.heartbeat_seconds) return;
    cout << "[fastSF] " << label << ": " << stage << ", lag idx=(" << x << "," << y << "," << z << ")" << endl;
    state.last_detail_time = now;
}

void print_q_stage(const std::string& label, const std::string& stage, ProgressState& state, int x, int y, int z, int q_order, bool force) {
    if (rank_mpi != 0) return;
    double now = MPI_Wtime();
    if (!force && (now - state.last_detail_time) < state.heartbeat_seconds) return;
    cout << "[fastSF] " << label << ": " << stage
         << ", q=" << q_order
         << ", lag idx=(" << x << "," << y << "," << z << ")" << endl;
    state.last_detail_time = now;
}

void print_run_summary() {
    if (rank_mpi != 0) return;
    cout << "[fastSF] Problem summary:" << endl;
    cout << "[fastSF]   field type: " << (scalar_switch ? "scalar" : "velocity") << endl;
    cout << "[fastSF]   dimensionality: " << (two_dimension_switch ? "2D" : "3D") << endl;
    if (two_dimension_switch) cout << "[fastSF]   grid: " << Nx << " x " << Nz << endl;
    else cout << "[fastSF]   grid: " << Nx << " x " << Ny << " x " << Nz << endl;
    cout << "[fastSF]   q range: " << q1 << " to " << q2 << endl;
    if (!scalar_switch) cout << "[fastSF]   longitudinal only: " << (longitudinal ? "true" : "false") << endl;
    cout << "[fastSF]   domain lengths: Lx=" << Lx << ", Ly=" << Ly << ", Lz=" << Lz << endl;
    cout << "[fastSF]   grid spacing: dx=" << dx << ", dy=" << dy << ", dz=" << dz << endl;
    cout << "[fastSF]   output directory: " << output_dir << endl;
}

namespace {

std::string path_parent(const std::string& path) {
    std::size_t slash = path.find_last_of("/\\");
    if (slash == std::string::npos) return "";
    return path.substr(0, slash);
}

std::string path_leaf(const std::string& path) {
    std::size_t slash = path.find_last_of("/\\");
    if (slash == std::string::npos) return path;
    return path.substr(slash + 1);
}

std::string join_path_local(const std::string& dir, const std::string& leaf) {
    if (dir.empty() || dir == ".") return leaf;
    if (leaf.empty()) return dir;
    if (dir[dir.size() - 1] == '/') return dir + leaf;
    return dir + "/" + leaf;
}

std::string trim_leading_zeros(const std::string& digits) {
    std::size_t first_nonzero = digits.find_first_not_of('0');
    if (first_nonzero == std::string::npos) return "0";
    return digits.substr(first_nonzero);
}

std::string cycle_label_from_input_path(const std::string& input_path) {
    std::string stem = basename_without_extension(input_path);
    int pos = static_cast<int>(stem.size()) - 1;
    while (pos >= 0 && std::isdigit(static_cast<unsigned char>(stem[pos]))) pos--;
    if (pos == static_cast<int>(stem.size()) - 1) return "out";
    return "structure_function_data_" + trim_leading_zeros(stem.substr(pos + 1));
}

}

void ensure_directory_exists(const std::string& dir_path) {
    if (dir_path.empty() || dir_path == ".") return;
    if (mkdir(dir_path.c_str(), 0777) != 0 && errno != EEXIST) {
        abort_with_error("Unable to create output directory '" + dir_path + "'");
    }
}

void configure_output_location(const std::string& input_path) {
    if (input_path.empty()) {
        output_dir = "out";
        return;
    }
    std::string input_dir = path_parent(input_path);
    std::string label = cycle_label_from_input_path(input_path);
    if (path_leaf(input_dir) == "in") output_dir = join_path_local(path_parent(input_dir), label);
    else if (!input_dir.empty()) output_dir = join_path_local(input_dir, label);
    else output_dir = label;
}

std::string output_file_path(const std::string& file_stem) {
    if (file_stem.find('/') != std::string::npos || file_stem.find('\\') != std::string::npos) return file_stem + ".h5";
    return join_path_local(output_dir, file_stem + ".h5");
}

bool h5_path_exists(hid_t object_id, const string& path) {
    return H5Lexists(object_id, path.c_str(), H5P_DEFAULT) > 0;
}

void write_string_attribute(hid_t object_id, const string& name, const string& value) {
    hid_t type_id = H5Tcopy(H5T_C_S1);
    H5Tset_size(type_id, value.size());
    hid_t space_id = H5Screate(H5S_SCALAR);
    hid_t attr_id = H5Acreate2(object_id, name.c_str(), type_id, space_id, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr_id, type_id, value.c_str());
    H5Aclose(attr_id);
    H5Sclose(space_id);
    H5Tclose(type_id);
}

void write_int_attribute(hid_t object_id, const string& name, int value) {
    hid_t space_id = H5Screate(H5S_SCALAR);
    hid_t attr_id = H5Acreate2(object_id, name.c_str(), H5T_NATIVE_INT, space_id, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr_id, H5T_NATIVE_INT, &value);
    H5Aclose(attr_id);
    H5Sclose(space_id);
}

void write_double_attribute(hid_t object_id, const string& name, double value) {
    hid_t space_id = H5Screate(H5S_SCALAR);
    hid_t attr_id = H5Acreate2(object_id, name.c_str(), H5T_NATIVE_DOUBLE, space_id, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr_id, H5T_NATIVE_DOUBLE, &value);
    H5Aclose(attr_id);
    H5Sclose(space_id);
}

bool read_bool_attribute(hid_t object_id, const std::string& name, bool default_value) {
    if (H5Aexists(object_id, name.c_str()) <= 0) return default_value;
    hid_t attr_id = H5Aopen(object_id, name.c_str(), H5P_DEFAULT);
    unsigned char value = default_value ? 1 : 0;
    if (H5Aread(attr_id, H5T_NATIVE_UCHAR, &value) < 0) {
        int int_value = default_value ? 1 : 0;
        H5Aread(attr_id, H5T_NATIVE_INT, &int_value);
        value = static_cast<unsigned char>(int_value != 0);
    }
    H5Aclose(attr_id);
    return value != 0;
}

void write_double_dataset_1d(hid_t file_id, const string& dataset_path, const std::vector<double>& values) {
    hsize_t dims[1] = {static_cast<hsize_t>(values.size())};
    hid_t space_id = H5Screate_simple(1, dims, NULL);
    hid_t dset_id = H5Dcreate2(file_id, dataset_path.c_str(), H5T_NATIVE_DOUBLE, space_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.empty() ? NULL : &values[0]);
    H5Dclose(dset_id);
    H5Sclose(space_id);
}

void write_double_dataset_3d(hid_t file_id, const string& dataset_path, const std::vector<double>& values, hsize_t nx, hsize_t ny, hsize_t nz) {
    hsize_t dims[3] = {nx, ny, nz};
    hid_t space_id = H5Screate_simple(3, dims, NULL);
    hid_t dset_id = H5Dcreate2(file_id, dataset_path.c_str(), H5T_NATIVE_DOUBLE, space_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.empty() ? NULL : &values[0]);
    H5Dclose(dset_id);
    H5Sclose(space_id);
}

std::vector<double> read_double_dataset_1d(hid_t file_id, const string& dataset_path) {
    hid_t dset_id = H5Dopen2(file_id, dataset_path.c_str(), H5P_DEFAULT);
    hid_t space_id = H5Dget_space(dset_id);
    hsize_t dims[1] = {0};
    H5Sget_simple_extent_dims(space_id, dims, NULL);
    std::vector<double> values(dims[0], 0.0);
    H5Dread(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.empty() ? NULL : &values[0]);
    H5Sclose(space_id);
    H5Dclose(dset_id);
    return values;
}

bool axis_has_periodic_duplicate(const std::vector<double>& axis, double domain_length) {
    if (axis.size() <= 1 || domain_length <= 0) return false;
    double spacing = axis[1] - axis[0];
    double tol = std::max(1.0, std::abs(domain_length)) * 1.0e-8;
    return std::abs((axis.back() - axis.front()) - domain_length) <= tol && spacing > 0;
}

std::vector<double> trim_periodic_axis(const std::vector<double>& axis, double domain_length) {
    if (!axis_has_periodic_duplicate(axis, domain_length)) return axis;
    return std::vector<double>(axis.begin(), axis.end() - 1);
}

bool is_structured_velocity_h5(const string& h5_path) {
    hid_t file_id = H5Fopen(h5_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) return false;
    bool ok = h5_path_exists(file_id, "/grid/x") && h5_path_exists(file_id, "/grid/y") && h5_path_exists(file_id, "/grid/z") &&
        h5_path_exists(file_id, "/fields/vx") && h5_path_exists(file_id, "/fields/vy") && h5_path_exists(file_id, "/fields/vz");
    H5Fclose(file_id);
    return ok;
}

bool is_structured_scalar_h5(const string& h5_path, string& field_path) {
    hid_t file_id = H5Fopen(h5_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) return false;
    bool ok = h5_path_exists(file_id, "/grid/x") && h5_path_exists(file_id, "/grid/y") && h5_path_exists(file_id, "/grid/z") &&
        h5_path_exists(file_id, "/fields");
    if (!ok) {
        H5Fclose(file_id);
        return false;
    }
    hid_t fields_group = H5Gopen2(file_id, "/fields", H5P_DEFAULT);
    hsize_t nobj = 0;
    H5Gget_num_objs(fields_group, &nobj);
    for (hsize_t i = 0; i < nobj; ++i) {
        char name_buf[256];
        ssize_t len = H5Gget_objname_by_idx(fields_group, i, name_buf, sizeof(name_buf));
        if (len <= 0) continue;
        string leaf(name_buf);
        if (leaf != "vx" && leaf != "vy" && leaf != "vz") {
            field_path = "/fields/" + leaf;
            H5Gclose(fields_group);
            H5Fclose(file_id);
            return true;
        }
    }
    H5Gclose(fields_group);
    H5Fclose(file_id);
    return false;
}

bool load_structured_grid_spacing(const string& h5_path) {
    hid_t file_id = H5Fopen(h5_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) return false;
    if (!(h5_path_exists(file_id, "/grid/x") && h5_path_exists(file_id, "/grid/y") && h5_path_exists(file_id, "/grid/z"))) {
        H5Fclose(file_id);
        return false;
    }
    std::vector<double> x_grid = read_double_dataset_1d(file_id, "/grid/x");
    std::vector<double> y_grid = read_double_dataset_1d(file_id, "/grid/y");
    std::vector<double> z_grid = read_double_dataset_1d(file_id, "/grid/z");
    bool periodic_duplicate_last = read_bool_attribute(file_id, "periodic_duplicate_last", false);
    H5Fclose(file_id);

    if (periodic_duplicate_last) {
        x_grid = trim_periodic_axis(x_grid, Lx);
        y_grid = trim_periodic_axis(y_grid, Ly);
        z_grid = trim_periodic_axis(z_grid, Lz);
    }

    Nx = static_cast<int>(x_grid.size());
    Ny = static_cast<int>(y_grid.size());
    Nz = static_cast<int>(z_grid.size());
    two_dimension_switch = (Ny <= 1);
    dx = validate_uniform_axis(x_grid, "x");
    dy = validate_uniform_axis(y_grid, "y");
    dz = validate_uniform_axis(z_grid, "z");
    Lx = (x_grid.size() > 1) ? (x_grid.back() - x_grid.front()) : 0.0;
    Ly = (y_grid.size() > 1) ? (y_grid.back() - y_grid.front()) : 0.0;
    Lz = (z_grid.size() > 1) ? (z_grid.back() - z_grid.front()) : 0.0;
    return true;
}

string default_scalar_field_path() {
    string leaf = dataset_leaf_name(TdName, "temp");
    if (leaf == "T.Fr") leaf = "temp";
    return "fields/" + leaf;
}

std::vector<std::pair<int,int> > split_axis_ranges(int length, int parts) {
    std::vector<std::pair<int,int> > ranges(parts);
    int base = (parts > 0) ? (length / parts) : 0;
    int remainder = (parts > 0) ? (length % parts) : 0;
    int start = 0;
    for (int rank = 0; rank < parts; ++rank) {
        int stop = start + base + (rank < remainder ? 1 : 0);
        ranges[rank] = std::make_pair(start, stop);
        start = stop;
    }
    return ranges;
}

void broadcast_txt_chunks(std::vector<TxtChunk>& chunks, int header_lines) {
    int chunk_count = (rank_mpi == 0) ? static_cast<int>(chunks.size()) : 0;
    MPI_Bcast(&chunk_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank_mpi != 0) chunks.resize(chunk_count);
    for (int i = 0; i < chunk_count; ++i) {
        long long offset = (rank_mpi == 0) ? chunks[i].offset : 0;
        long count = (rank_mpi == 0) ? chunks[i].count : 0;
        MPI_Bcast(&offset, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
        MPI_Bcast(&count, 1, MPI_LONG, 0, MPI_COMM_WORLD);
        if (rank_mpi != 0) {
            chunks[i].offset = offset;
            chunks[i].count = count;
        }
    }
}

std::vector<double> gather_unique_axis_values(const std::set<double>& local_values) {
    std::vector<double> local(local_values.begin(), local_values.end());
    int local_count = static_cast<int>(local.size());
    std::vector<int> counts;
    if (rank_mpi == 0) counts.resize(P, 0);
    MPI_Gather(&local_count, 1, MPI_INT, rank_mpi == 0 ? &counts[0] : NULL, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> displs;
    std::vector<double> gathered;
    if (rank_mpi == 0) {
        displs.resize(P, 0);
        int total = 0;
        for (int i = 0; i < P; ++i) {
            displs[i] = total;
            total += counts[i];
        }
        gathered.resize(total, 0.0);
    }

    MPI_Gatherv(local.empty() ? NULL : &local[0], local_count, MPI_DOUBLE,
        rank_mpi == 0 && !gathered.empty() ? &gathered[0] : NULL,
        rank_mpi == 0 ? &counts[0] : NULL,
        rank_mpi == 0 ? &displs[0] : NULL,
        MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<double> merged;
    if (rank_mpi == 0) {
        std::sort(gathered.begin(), gathered.end());
        for (std::size_t i = 0; i < gathered.size(); ++i) {
            if (merged.empty() || std::abs(gathered[i] - merged.back()) > 1.0e-12) merged.push_back(gathered[i]);
        }
    }

    int merged_count = (rank_mpi == 0) ? static_cast<int>(merged.size()) : 0;
    MPI_Bcast(&merged_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank_mpi != 0) merged.resize(merged_count, 0.0);
    if (merged_count > 0) MPI_Bcast(&merged[0], merged_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return merged;
}

int owner_rank_for_x_index(int ix, const std::vector<std::pair<int,int> >& x_ranges) {
    for (int rank = 0; rank < static_cast<int>(x_ranges.size()); ++rank) {
        if (ix >= x_ranges[rank].first && ix < x_ranges[rank].second) return rank;
    }
    return static_cast<int>(x_ranges.size()) - 1;
}

void write_parallel_velocity_h5(const string& h5_path, const StructuredGridInfo& grid, int x_start, int x_stop, const std::vector<double>& local_vx, const std::vector<double>& local_vy, const std::vector<double>& local_vz) {
    print_stage("Writing structured velocity HDF5 fields in parallel...");
    hid_t fapl_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(fapl_id, MPI_COMM_WORLD, MPI_INFO_NULL);
    hid_t file_id = H5Fcreate(h5_path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);
    H5Pclose(fapl_id);

    hid_t fields_group = H5Gcreate2(file_id, "/fields", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Gclose(fields_group);

    hsize_t dims[3] = {static_cast<hsize_t>(grid.x.size()), static_cast<hsize_t>(grid.y.size()), static_cast<hsize_t>(grid.z.size())};
    hid_t file_space = H5Screate_simple(3, dims, NULL);
    hid_t vx_dset = H5Dcreate2(file_id, "/fields/vx", H5T_NATIVE_DOUBLE, file_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t vy_dset = H5Dcreate2(file_id, "/fields/vy", H5T_NATIVE_DOUBLE, file_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t vz_dset = H5Dcreate2(file_id, "/fields/vz", H5T_NATIVE_DOUBLE, file_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(file_space);

    int local_nx = x_stop - x_start;
    hid_t dxpl_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(dxpl_id, H5FD_MPIO_COLLECTIVE);

    hid_t vx_file_space = H5Dget_space(vx_dset);
    hid_t vy_file_space = H5Dget_space(vy_dset);
    hid_t vz_file_space = H5Dget_space(vz_dset);
    hid_t mem_space = H5Screate(H5S_NULL);
    if (local_nx > 0) {
        hsize_t start[3] = {static_cast<hsize_t>(x_start), 0, 0};
        hsize_t count[3] = {static_cast<hsize_t>(local_nx), static_cast<hsize_t>(grid.y.size()), static_cast<hsize_t>(grid.z.size())};
        H5Sselect_hyperslab(vx_file_space, H5S_SELECT_SET, start, NULL, count, NULL);
        H5Sselect_hyperslab(vy_file_space, H5S_SELECT_SET, start, NULL, count, NULL);
        H5Sselect_hyperslab(vz_file_space, H5S_SELECT_SET, start, NULL, count, NULL);
        mem_space = H5Screate_simple(3, count, NULL);
    } else {
        H5Sselect_none(vx_file_space);
        H5Sselect_none(vy_file_space);
        H5Sselect_none(vz_file_space);
    }

    H5Dwrite(vx_dset, H5T_NATIVE_DOUBLE, mem_space, vx_file_space, dxpl_id, local_nx > 0 ? &local_vx[0] : NULL);
    H5Dwrite(vy_dset, H5T_NATIVE_DOUBLE, mem_space, vy_file_space, dxpl_id, local_nx > 0 ? &local_vy[0] : NULL);
    H5Dwrite(vz_dset, H5T_NATIVE_DOUBLE, mem_space, vz_file_space, dxpl_id, local_nx > 0 ? &local_vz[0] : NULL);

    H5Sclose(mem_space);
    H5Sclose(vx_file_space);
    H5Sclose(vy_file_space);
    H5Sclose(vz_file_space);
    H5Pclose(dxpl_id);
    H5Dclose(vx_dset);
    H5Dclose(vy_dset);
    H5Dclose(vz_dset);
    H5Fclose(file_id);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank_mpi == 0) {
        print_stage("Writing structured velocity grid metadata...");
        hid_t serial_file = H5Fopen(h5_path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
        hid_t grid_group = H5Gcreate2(serial_file, "/grid", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Gclose(grid_group);
        write_double_dataset_1d(serial_file, "/grid/x", grid.x);
        write_double_dataset_1d(serial_file, "/grid/y", grid.y);
        write_double_dataset_1d(serial_file, "/grid/z", grid.z);
        write_string_attribute(serial_file, "schema", "structured_velocity_v1");
        write_int_attribute(serial_file, "periodic_duplicate_last", 0);
        write_int_attribute(serial_file, "Nx", static_cast<int>(grid.x.size()));
        write_int_attribute(serial_file, "Ny", static_cast<int>(grid.y.size()));
        write_int_attribute(serial_file, "Nz", static_cast<int>(grid.z.size()));
        write_int_attribute(serial_file, "fft_nx", static_cast<int>(grid.x.size()));
        write_int_attribute(serial_file, "fft_ny", static_cast<int>(grid.y.size()));
        write_int_attribute(serial_file, "fft_nz", static_cast<int>(grid.z.size()));
        write_double_attribute(serial_file, "dx", grid.dx);
        write_double_attribute(serial_file, "dy", grid.dy);
        write_double_attribute(serial_file, "dz", grid.dz);
        H5Fclose(serial_file);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

void write_parallel_scalar_h5(const string& h5_path, const string& field_path, const StructuredGridInfo& grid, int x_start, int x_stop, const std::vector<double>& local_field) {
    string dataset_path = "/" + field_path;
    print_stage("Writing structured scalar HDF5 field in parallel...");
    hid_t fapl_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(fapl_id, MPI_COMM_WORLD, MPI_INFO_NULL);
    hid_t file_id = H5Fcreate(h5_path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);
    H5Pclose(fapl_id);

    hid_t fields_group = H5Gcreate2(file_id, "/fields", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Gclose(fields_group);

    hsize_t dims[3] = {static_cast<hsize_t>(grid.x.size()), static_cast<hsize_t>(grid.y.size()), static_cast<hsize_t>(grid.z.size())};
    hid_t file_space = H5Screate_simple(3, dims, NULL);
    hid_t field_dset = H5Dcreate2(file_id, dataset_path.c_str(), H5T_NATIVE_DOUBLE, file_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(file_space);

    int local_nx = x_stop - x_start;
    hid_t dxpl_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(dxpl_id, H5FD_MPIO_COLLECTIVE);

    hid_t field_file_space = H5Dget_space(field_dset);
    hid_t mem_space = H5Screate(H5S_NULL);
    if (local_nx > 0) {
        hsize_t start[3] = {static_cast<hsize_t>(x_start), 0, 0};
        hsize_t count[3] = {static_cast<hsize_t>(local_nx), static_cast<hsize_t>(grid.y.size()), static_cast<hsize_t>(grid.z.size())};
        H5Sselect_hyperslab(field_file_space, H5S_SELECT_SET, start, NULL, count, NULL);
        mem_space = H5Screate_simple(3, count, NULL);
    } else {
        H5Sselect_none(field_file_space);
    }

    H5Dwrite(field_dset, H5T_NATIVE_DOUBLE, mem_space, field_file_space, dxpl_id, local_nx > 0 ? &local_field[0] : NULL);

    H5Sclose(mem_space);
    H5Sclose(field_file_space);
    H5Pclose(dxpl_id);
    H5Dclose(field_dset);
    H5Fclose(file_id);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank_mpi == 0) {
        print_stage("Writing structured scalar grid metadata...");
        hid_t serial_file = H5Fopen(h5_path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
        hid_t grid_group = H5Gcreate2(serial_file, "/grid", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Gclose(grid_group);
        write_double_dataset_1d(serial_file, "/grid/x", grid.x);
        write_double_dataset_1d(serial_file, "/grid/y", grid.y);
        write_double_dataset_1d(serial_file, "/grid/z", grid.z);
        write_string_attribute(serial_file, "schema", "structured_scalar_v1");
        write_int_attribute(serial_file, "periodic_duplicate_last", 0);
        write_int_attribute(serial_file, "Nx", static_cast<int>(grid.x.size()));
        write_int_attribute(serial_file, "Ny", static_cast<int>(grid.y.size()));
        write_int_attribute(serial_file, "Nz", static_cast<int>(grid.z.size()));
        write_int_attribute(serial_file, "fft_nx", static_cast<int>(grid.x.size()));
        write_int_attribute(serial_file, "fft_ny", static_cast<int>(grid.y.size()));
        write_int_attribute(serial_file, "fft_nz", static_cast<int>(grid.z.size()));
        write_double_attribute(serial_file, "dx", grid.dx);
        write_double_attribute(serial_file, "dy", grid.dy);
        write_double_attribute(serial_file, "dz", grid.dz);
        H5Fclose(serial_file);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

void convert_velocity_txt_to_structured_h5(const string& txt_path, const string& h5_path) {
    if (rank_mpi == 0) cout << "Detected TXT: " << txt_path << ". Converting to HDF5..." << endl;
    print_stage("Stage 1/5: analyzing sampled-data header and chunk index...");

    int header_lines = 0;
    std::vector<TxtChunk> chunks;
    if (rank_mpi == 0) {
        try {
            std::pair<std::vector<std::string>, int> header = detect_txt_header(txt_path);
            header_lines = header.second;
            chunks = build_txt_chunk_index(txt_path, header_lines);
            if (chunks.empty()) abort_parallel_conversion("No sampled-data rows found in '" + txt_path + "'");
        } catch (const std::exception& err) {
            abort_parallel_conversion(err.what());
        }
    }
    MPI_Bcast(&header_lines, 1, MPI_INT, 0, MPI_COMM_WORLD);
    broadcast_txt_chunks(chunks, header_lines);

    std::set<double> local_x_set, local_y_set, local_z_set;
    long local_rows = 0;
    try {
        for (int ci = rank_mpi; ci < static_cast<int>(chunks.size()); ci += P) {
            SampledTxtData chunk = read_txt_chunk(txt_path, chunks[ci].offset, chunks[ci].count, false);
            local_rows += static_cast<long>(chunk.v1.size());
            for (std::size_t j = 0; j < chunk.x.size(); ++j) {
                local_x_set.insert(std::round(chunk.x[j] * 1.0e10) / 1.0e10);
                local_y_set.insert(std::round(chunk.y[j] * 1.0e10) / 1.0e10);
                local_z_set.insert(std::round(chunk.z[j] * 1.0e10) / 1.0e10);
            }
        }
    } catch (const std::exception& err) {
        abort_parallel_conversion(err.what());
    }

    StructuredGridInfo full_grid, grid;
    try {
        print_stage("Stage 2/5: discovering structured grid from sampled-data coordinates...");
        full_grid.x = gather_unique_axis_values(local_x_set);
        full_grid.y = gather_unique_axis_values(local_y_set);
        full_grid.z = gather_unique_axis_values(local_z_set);
        full_grid.dx = validate_uniform_axis(full_grid.x, "x");
        full_grid.dy = validate_uniform_axis(full_grid.y, "y");
        full_grid.dz = validate_uniform_axis(full_grid.z, "z");
        grid.x = trim_periodic_axis(full_grid.x, Lx);
        grid.y = trim_periodic_axis(full_grid.y, Ly);
        grid.z = trim_periodic_axis(full_grid.z, Lz);
        grid.dx = validate_uniform_axis(grid.x, "x");
        grid.dy = validate_uniform_axis(grid.y, "y");
        grid.dz = validate_uniform_axis(grid.z, "z");
        if (grid.y.size() <= 1) abort_parallel_conversion("Sampled-data TXT conversion currently supports only 3D inputs");
    } catch (const std::exception& err) {
        abort_parallel_conversion(err.what());
    }
    int trim_last_x = (grid.x.size() + 1 == full_grid.x.size()) ? 1 : 0;
    int trim_last_y = (grid.y.size() + 1 == full_grid.y.size()) ? 1 : 0;
    int trim_last_z = (grid.z.size() + 1 == full_grid.z.size()) ? 1 : 0;
    if (rank_mpi == 0) {
        cout << "[fastSF] Structured sampled-data grid: " << grid.x.size() << " x " << grid.y.size() << " x " << grid.z.size() << endl;
        if (trim_last_x || trim_last_y || trim_last_z) {
            cout << "[fastSF] Removed duplicated periodic endpoint planes: "
                 << "x=" << trim_last_x << ", y=" << trim_last_y << ", z=" << trim_last_z << endl;
        }
    }

    long total_rows = 0;
    MPI_Allreduce(&local_rows, &total_rows, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    long full_rows = static_cast<long>(full_grid.x.size() * full_grid.y.size() * full_grid.z.size());
    if (full_rows != total_rows) abort_parallel_conversion("Structured grid size does not match sampled-data row count for '" + txt_path + "'");
    long expected_rows = static_cast<long>(grid.x.size() * grid.y.size() * grid.z.size());

    std::vector<std::pair<int,int> > x_ranges = split_axis_ranges(static_cast<int>(grid.x.size()), P);
    int x_start = x_ranges[rank_mpi].first;
    int x_stop = x_ranges[rank_mpi].second;
    int local_nx = x_stop - x_start;
    int ny = static_cast<int>(grid.y.size());
    int nz = static_cast<int>(grid.z.size());

    std::vector< std::vector<double> > send_bins(P);
    try {
        print_stage("Stage 3/5: redistributing sampled-data rows into MPI x-slabs...");
        for (int ci = rank_mpi; ci < static_cast<int>(chunks.size()); ci += P) {
            SampledTxtData chunk = read_txt_chunk(txt_path, chunks[ci].offset, chunks[ci].count, false);
            std::vector<int> ix = compute_axis_indices(chunk.x, full_grid.x, full_grid.dx, "x");
            std::vector<int> iy = compute_axis_indices(chunk.y, full_grid.y, full_grid.dy, "y");
            std::vector<int> iz = compute_axis_indices(chunk.z, full_grid.z, full_grid.dz, "z");
            for (std::size_t row = 0; row < chunk.v1.size(); ++row) {
                if ((trim_last_x && ix[row] == static_cast<int>(full_grid.x.size()) - 1) ||
                    (trim_last_y && iy[row] == static_cast<int>(full_grid.y.size()) - 1) ||
                    (trim_last_z && iz[row] == static_cast<int>(full_grid.z.size()) - 1)) continue;
                int dest = owner_rank_for_x_index(ix[row], x_ranges);
                send_bins[dest].push_back(static_cast<double>(ix[row]));
                send_bins[dest].push_back(static_cast<double>(iy[row]));
                send_bins[dest].push_back(static_cast<double>(iz[row]));
                send_bins[dest].push_back(chunk.v1[row]);
                send_bins[dest].push_back(chunk.v2[row]);
                send_bins[dest].push_back(chunk.v3[row]);
            }
        }
    } catch (const std::exception& err) {
        abort_parallel_conversion(err.what());
    }

    std::vector<int> send_counts(P, 0), recv_counts(P, 0), send_displs(P, 0), recv_displs(P, 0);
    for (int i = 0; i < P; ++i) send_counts[i] = static_cast<int>(send_bins[i].size());
    MPI_Alltoall(&send_counts[0], 1, MPI_INT, &recv_counts[0], 1, MPI_INT, MPI_COMM_WORLD);
    for (int i = 1; i < P; ++i) {
        send_displs[i] = send_displs[i - 1] + send_counts[i - 1];
        recv_displs[i] = recv_displs[i - 1] + recv_counts[i - 1];
    }

    std::vector<double> sendbuf;
    sendbuf.reserve(send_displs[P - 1] + send_counts[P - 1]);
    for (int i = 0; i < P; ++i) sendbuf.insert(sendbuf.end(), send_bins[i].begin(), send_bins[i].end());
    std::vector<double> recvbuf(recv_displs[P - 1] + recv_counts[P - 1], 0.0);
    MPI_Alltoallv(sendbuf.empty() ? NULL : &sendbuf[0], &send_counts[0], &send_displs[0], MPI_DOUBLE,
        recvbuf.empty() ? NULL : &recvbuf[0], &recv_counts[0], &recv_displs[0], MPI_DOUBLE, MPI_COMM_WORLD);

    std::vector<double> local_vx(static_cast<std::size_t>(local_nx) * ny * nz, 0.0);
    std::vector<double> local_vy(static_cast<std::size_t>(local_nx) * ny * nz, 0.0);
    std::vector<double> local_vz(static_cast<std::size_t>(local_nx) * ny * nz, 0.0);
    std::vector<char> filled(static_cast<std::size_t>(local_nx) * ny * nz, 0);
    try {
        for (std::size_t pos = 0; pos < recvbuf.size(); pos += 6) {
            int ix = static_cast<int>(recvbuf[pos + 0]);
            int iy = static_cast<int>(recvbuf[pos + 1]);
            int iz = static_cast<int>(recvbuf[pos + 2]);
            int local_ix = ix - x_start;
            long flat = (static_cast<long>(local_ix) * ny + iy) * nz + iz;
            if (local_ix < 0 || local_ix >= local_nx || iy < 0 || iy >= ny || iz < 0 || iz >= nz) abort_parallel_conversion("Redistributed sampled-data row mapped out of bounds");
            if (filled[flat]) abort_parallel_conversion("Duplicate sampled-data coordinate detected in '" + txt_path + "'");
            filled[flat] = 1;
            local_vx[flat] = recvbuf[pos + 3];
            local_vy[flat] = recvbuf[pos + 4];
            local_vz[flat] = recvbuf[pos + 5];
        }
    } catch (const std::exception& err) {
        abort_parallel_conversion(err.what());
    }
    if (!filled.empty() && static_cast<std::size_t>(std::count(filled.begin(), filled.end(), 1)) != filled.size()) {
        abort_parallel_conversion("Rank did not receive a complete x-slab during sampled-data redistribution");
    }

    print_stage("Stage 4/5: assembling local velocity slab and writing HDF5...");
    write_parallel_velocity_h5(h5_path, grid, x_start, x_stop, local_vx, local_vy, local_vz);
    std::vector< std::vector<double> >().swap(send_bins);
    std::vector<double>().swap(sendbuf);
    std::vector<double>().swap(recvbuf);
    std::vector<double>().swap(local_vx);
    std::vector<double>().swap(local_vy);
    std::vector<double>().swap(local_vz);
    std::vector<char>().swap(filled);

    if (rank_mpi == 0) {
        print_stage("Stage 5/5: verifying converted velocity HDF5...");
        hid_t verify_file = H5Fopen(h5_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        hid_t verify_ds = H5Dopen2(verify_file, "/fields/vx", H5P_DEFAULT);
        hid_t verify_space = H5Dget_space(verify_ds);
        hsize_t dims[3] = {0, 0, 0};
        H5Sget_simple_extent_dims(verify_space, dims, NULL);
        H5Sclose(verify_space);
        H5Dclose(verify_ds);
        H5Fclose(verify_file);
        if (static_cast<long>(dims[0] * dims[1] * dims[2]) != expected_rows) abort_parallel_conversion("Verification failed for converted HDF5 '" + h5_path + "'");
        cout << "  SUCCESS: Conversion verified. Deleting " << txt_path << endl;
        std::remove(txt_path.c_str());
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

void convert_scalar_txt_to_structured_h5(const string& txt_path, const string& h5_path, const string& field_path) {
    if (rank_mpi == 0) cout << "Detected TXT: " << txt_path << ". Converting to HDF5..." << endl;
    print_stage("Stage 1/5: analyzing sampled scalar header and chunk index...");

    int header_lines = 0;
    std::vector<TxtChunk> chunks;
    if (rank_mpi == 0) {
        try {
            std::pair<std::vector<std::string>, int> header = detect_txt_header(txt_path);
            header_lines = header.second;
            chunks = build_txt_chunk_index(txt_path, header_lines);
            if (chunks.empty()) abort_parallel_conversion("No sampled-data rows found in '" + txt_path + "'");
        } catch (const std::exception& err) {
            abort_parallel_conversion(err.what());
        }
    }
    MPI_Bcast(&header_lines, 1, MPI_INT, 0, MPI_COMM_WORLD);
    broadcast_txt_chunks(chunks, header_lines);

    std::set<double> local_x_set, local_y_set, local_z_set;
    long local_rows = 0;
    try {
        for (int ci = rank_mpi; ci < static_cast<int>(chunks.size()); ci += P) {
            SampledTxtData chunk = read_txt_chunk(txt_path, chunks[ci].offset, chunks[ci].count, true);
            local_rows += static_cast<long>(chunk.v1.size());
            for (std::size_t j = 0; j < chunk.x.size(); ++j) {
                local_x_set.insert(std::round(chunk.x[j] * 1.0e10) / 1.0e10);
                local_y_set.insert(std::round(chunk.y[j] * 1.0e10) / 1.0e10);
                local_z_set.insert(std::round(chunk.z[j] * 1.0e10) / 1.0e10);
            }
        }
    } catch (const std::exception& err) {
        abort_parallel_conversion(err.what());
    }

    StructuredGridInfo full_grid, grid;
    try {
        print_stage("Stage 2/5: discovering structured scalar grid from sampled-data coordinates...");
        full_grid.x = gather_unique_axis_values(local_x_set);
        full_grid.y = gather_unique_axis_values(local_y_set);
        full_grid.z = gather_unique_axis_values(local_z_set);
        full_grid.dx = validate_uniform_axis(full_grid.x, "x");
        full_grid.dy = validate_uniform_axis(full_grid.y, "y");
        full_grid.dz = validate_uniform_axis(full_grid.z, "z");
        grid.x = trim_periodic_axis(full_grid.x, Lx);
        grid.y = trim_periodic_axis(full_grid.y, Ly);
        grid.z = trim_periodic_axis(full_grid.z, Lz);
        grid.dx = validate_uniform_axis(grid.x, "x");
        grid.dy = validate_uniform_axis(grid.y, "y");
        grid.dz = validate_uniform_axis(grid.z, "z");
        if (grid.y.size() <= 1) abort_parallel_conversion("Sampled-data TXT conversion currently supports only 3D inputs");
    } catch (const std::exception& err) {
        abort_parallel_conversion(err.what());
    }
    int trim_last_x = (grid.x.size() + 1 == full_grid.x.size()) ? 1 : 0;
    int trim_last_y = (grid.y.size() + 1 == full_grid.y.size()) ? 1 : 0;
    int trim_last_z = (grid.z.size() + 1 == full_grid.z.size()) ? 1 : 0;
    if (rank_mpi == 0) {
        cout << "[fastSF] Structured scalar grid: " << grid.x.size() << " x " << grid.y.size() << " x " << grid.z.size() << endl;
        if (trim_last_x || trim_last_y || trim_last_z) {
            cout << "[fastSF] Removed duplicated periodic endpoint planes: "
                 << "x=" << trim_last_x << ", y=" << trim_last_y << ", z=" << trim_last_z << endl;
        }
    }

    long total_rows = 0;
    MPI_Allreduce(&local_rows, &total_rows, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    long full_rows = static_cast<long>(full_grid.x.size() * full_grid.y.size() * full_grid.z.size());
    if (full_rows != total_rows) abort_parallel_conversion("Structured grid size does not match sampled-data row count for '" + txt_path + "'");
    long expected_rows = static_cast<long>(grid.x.size() * grid.y.size() * grid.z.size());

    std::vector<std::pair<int,int> > x_ranges = split_axis_ranges(static_cast<int>(grid.x.size()), P);
    int x_start = x_ranges[rank_mpi].first;
    int x_stop = x_ranges[rank_mpi].second;
    int local_nx = x_stop - x_start;
    int ny = static_cast<int>(grid.y.size());
    int nz = static_cast<int>(grid.z.size());

    std::vector< std::vector<double> > send_bins(P);
    try {
        print_stage("Stage 3/5: redistributing sampled scalar rows into MPI x-slabs...");
        for (int ci = rank_mpi; ci < static_cast<int>(chunks.size()); ci += P) {
            SampledTxtData chunk = read_txt_chunk(txt_path, chunks[ci].offset, chunks[ci].count, true);
            std::vector<int> ix = compute_axis_indices(chunk.x, full_grid.x, full_grid.dx, "x");
            std::vector<int> iy = compute_axis_indices(chunk.y, full_grid.y, full_grid.dy, "y");
            std::vector<int> iz = compute_axis_indices(chunk.z, full_grid.z, full_grid.dz, "z");
            for (std::size_t row = 0; row < chunk.v1.size(); ++row) {
                if ((trim_last_x && ix[row] == static_cast<int>(full_grid.x.size()) - 1) ||
                    (trim_last_y && iy[row] == static_cast<int>(full_grid.y.size()) - 1) ||
                    (trim_last_z && iz[row] == static_cast<int>(full_grid.z.size()) - 1)) continue;
                int dest = owner_rank_for_x_index(ix[row], x_ranges);
                send_bins[dest].push_back(static_cast<double>(ix[row]));
                send_bins[dest].push_back(static_cast<double>(iy[row]));
                send_bins[dest].push_back(static_cast<double>(iz[row]));
                send_bins[dest].push_back(chunk.v1[row]);
            }
        }
    } catch (const std::exception& err) {
        abort_parallel_conversion(err.what());
    }

    std::vector<int> send_counts(P, 0), recv_counts(P, 0), send_displs(P, 0), recv_displs(P, 0);
    for (int i = 0; i < P; ++i) send_counts[i] = static_cast<int>(send_bins[i].size());
    MPI_Alltoall(&send_counts[0], 1, MPI_INT, &recv_counts[0], 1, MPI_INT, MPI_COMM_WORLD);
    for (int i = 1; i < P; ++i) {
        send_displs[i] = send_displs[i - 1] + send_counts[i - 1];
        recv_displs[i] = recv_displs[i - 1] + recv_counts[i - 1];
    }

    std::vector<double> sendbuf;
    sendbuf.reserve(send_displs[P - 1] + send_counts[P - 1]);
    for (int i = 0; i < P; ++i) sendbuf.insert(sendbuf.end(), send_bins[i].begin(), send_bins[i].end());
    std::vector<double> recvbuf(recv_displs[P - 1] + recv_counts[P - 1], 0.0);
    MPI_Alltoallv(sendbuf.empty() ? NULL : &sendbuf[0], &send_counts[0], &send_displs[0], MPI_DOUBLE,
        recvbuf.empty() ? NULL : &recvbuf[0], &recv_counts[0], &recv_displs[0], MPI_DOUBLE, MPI_COMM_WORLD);

    std::vector<double> local_field(static_cast<std::size_t>(local_nx) * ny * nz, 0.0);
    std::vector<char> filled(static_cast<std::size_t>(local_nx) * ny * nz, 0);
    for (std::size_t pos = 0; pos < recvbuf.size(); pos += 4) {
        int ix = static_cast<int>(recvbuf[pos + 0]);
        int iy = static_cast<int>(recvbuf[pos + 1]);
        int iz = static_cast<int>(recvbuf[pos + 2]);
        int local_ix = ix - x_start;
        long flat = (static_cast<long>(local_ix) * ny + iy) * nz + iz;
        if (local_ix < 0 || local_ix >= local_nx || iy < 0 || iy >= ny || iz < 0 || iz >= nz) abort_parallel_conversion("Redistributed scalar sampled-data row mapped out of bounds");
        if (filled[flat]) abort_parallel_conversion("Duplicate sampled-data coordinate detected in '" + txt_path + "'");
        filled[flat] = 1;
        local_field[flat] = recvbuf[pos + 3];
    }
    if (!filled.empty() && static_cast<std::size_t>(std::count(filled.begin(), filled.end(), 1)) != filled.size()) {
        abort_parallel_conversion("Rank did not receive a complete x-slab during scalar sampled-data redistribution");
    }

    print_stage("Stage 4/5: assembling local scalar slab and writing HDF5...");
    write_parallel_scalar_h5(h5_path, field_path, grid, x_start, x_stop, local_field);
    std::vector< std::vector<double> >().swap(send_bins);
    std::vector<double>().swap(sendbuf);
    std::vector<double>().swap(recvbuf);
    std::vector<double>().swap(local_field);
    std::vector<char>().swap(filled);

    if (rank_mpi == 0) {
        print_stage("Stage 5/5: verifying converted scalar HDF5...");
        hid_t verify_file = H5Fopen(h5_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        hid_t verify_ds = H5Dopen2(verify_file, ("/" + field_path).c_str(), H5P_DEFAULT);
        hid_t verify_space = H5Dget_space(verify_ds);
        hsize_t dims[3] = {0, 0, 0};
        H5Sget_simple_extent_dims(verify_space, dims, NULL);
        H5Sclose(verify_space);
        H5Dclose(verify_ds);
        H5Fclose(verify_file);
        if (static_cast<long>(dims[0] * dims[1] * dims[2]) != expected_rows) abort_parallel_conversion("Verification failed for converted HDF5 '" + h5_path + "'");
        cout << "  SUCCESS: Conversion verified. Deleting " << txt_path << endl;
        std::remove(txt_path.c_str());
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

void prepare_scalar_input() {
    print_stage("Resolving scalar input source...");
    std::vector<std::string> search_dirs;
    search_dirs.push_back("");
    search_dirs.push_back("in");
    search_dirs.push_back("data");
    ResolvedInputPath source = resolve_input_path(TName, search_dirs);
    configure_output_location(source.full_path);
    if (source.extension == ".txt") {
        string field_path = default_scalar_field_path();
        convert_scalar_txt_to_structured_h5(source.full_path, source.path_without_extension + ".h5", field_path);
        TName = source.path_without_extension;
        TdName = field_path;
        return;
    }
    print_stage("Using direct scalar HDF5 input.");
    TName = source.path_without_extension;
    string detected_field;
    if (is_structured_scalar_h5(source.full_path, detected_field) && TdName == "T.Fr") TdName = detected_field.substr(1);
}

void prepare_velocity_inputs() {
    print_stage("Resolving velocity input source...");
    bool shared_velocity_file = (UName == VName && UName == WName);
    if (shared_velocity_file) {
        std::vector<std::string> search_dirs;
        search_dirs.push_back("");
        search_dirs.push_back("in");
        search_dirs.push_back("data");
        ResolvedInputPath source = resolve_input_path(UName, search_dirs);
        configure_output_location(source.full_path);
        if (source.extension == ".txt") {
            convert_velocity_txt_to_structured_h5(source.full_path, source.path_without_extension + ".h5");
            UName = source.path_without_extension;
            VName = source.path_without_extension;
            WName = source.path_without_extension;
            UdName = "fields/vx";
            VdName = "fields/vy";
            WdName = "fields/vz";
            return;
        }
        print_stage("Using direct structured velocity HDF5 input.");
        UName = source.path_without_extension;
        VName = source.path_without_extension;
        WName = source.path_without_extension;
        if (is_structured_velocity_h5(source.full_path) && UdName == "U.V1r" && VdName == "U.V2r" && WdName == "U.V3r") {
            UdName = "fields/vx";
            VdName = "fields/vy";
            WdName = "fields/vz";
        }
        return;
    }

    std::vector<std::string> search_dirs;
    search_dirs.push_back("");
    search_dirs.push_back("in");
    search_dirs.push_back("data");
    ResolvedInputPath u_source = resolve_input_path(UName, search_dirs);
    ResolvedInputPath v_source = resolve_input_path(VName, search_dirs);
    ResolvedInputPath w_source = resolve_input_path(WName, search_dirs);
    configure_output_location(u_source.full_path);
    if (u_source.extension == ".txt" || v_source.extension == ".txt" || w_source.extension == ".txt") {
        throw std::runtime_error("Velocity TXT input must be provided as one sampled-data file, matching the turbulence_post_process format");
    }
    print_stage("Using legacy multi-file HDF5 velocity input.");
    UName = u_source.path_without_extension;
    VName = v_source.path_without_extension;
    WName = w_source.path_without_extension;
}

void prepare_input_sources() {
    try {
        if (scalar_switch) prepare_scalar_input();
        else prepare_velocity_inputs();
    } catch (const std::exception& err) {
        abort_with_error(err.what());
    }
}

bool str_to_bool(string s){
	if (s=="true" || s=="1") return true;
	if (s=="false" || s=="0") return false;
    if (rank_mpi==0) cout<<"Invalid input\n";
	exit(1);
}

void get_input_shape(string fold, string file, string dset, Array<int,1>& s){
	ifstream f_chk(fold+file+".h5");
	s.resize(4);
	if (f_chk.is_open()){
    	f_chk.close();
        hid_t file_id = H5Fopen((fold+file+".h5").c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        bool structured = h5_path_exists(file_id, "/grid/x") && h5_path_exists(file_id, "/grid/y") && h5_path_exists(file_id, "/grid/z");
        if (structured) {
            std::vector<double> x_grid = read_double_dataset_1d(file_id, "/grid/x");
            std::vector<double> y_grid = read_double_dataset_1d(file_id, "/grid/y");
            std::vector<double> z_grid = read_double_dataset_1d(file_id, "/grid/z");
            bool periodic_duplicate_last = read_bool_attribute(file_id, "periodic_duplicate_last", false);
            if (periodic_duplicate_last) {
                x_grid = trim_periodic_axis(x_grid, Lx);
                y_grid = trim_periodic_axis(y_grid, Ly);
                z_grid = trim_periodic_axis(z_grid, Lz);
            }
            Nx = static_cast<int>(x_grid.size());
            Ny = static_cast<int>(y_grid.size());
            Nz = static_cast<int>(z_grid.size());
            two_dimension_switch = (Ny <= 1);
            s(0)=two_dimension_switch ? 2 : 3; s(1)=Nx; s(2)=Ny; s(3)=Nz;
        } else {
            h5::File f(fold+file+".h5", "r");
            h5::Dataset ds = f[dset];
            int dim = ds.shape().size();
            if (dim==2){ two_dimension_switch=true; Nx=ds.shape()[0]; Ny=1; Nz=ds.shape()[1]; }
            else if (dim==3 && ds.shape()[1] == 1) { two_dimension_switch=true; Nx=ds.shape()[0]; Ny=1; Nz=ds.shape()[2]; }
            else { two_dimension_switch=false; Nx=ds.shape()[0]; Ny=ds.shape()[1]; Nz=ds.shape()[2]; }
            s(0)=dim; s(1)=Nx; s(2)=Ny; s(3)=Nz;
        }
        H5Fclose(file_id);
  	} else {
    	if (rank_mpi==0) cerr<<"\nDesired file "<<fold+file+".h5 does not exist\n\n";
    	h5::finalize(); finalize_mpi_runtime(); exit(1);
    }
}

void resize_input(){
	if(two_dimension_switch) scalar_switch ? T_2D.resize(Nx, Nz) : (V1_2D.resize(Nx, Nz), V3_2D.resize(Nx, Nz));
    else scalar_switch ? T.resize(Nx, Ny, Nz) : (V1.resize(Nx,Ny,Nz), V2.resize(Nx,Ny,Nz), V3.resize(Nx,Ny,Nz));
}

bool compare(Array<int,1> A, Array<int,1> B){
	for (int i=0; i<A.size(); i++) if (A(i)!=B(i)) return false;
	return true;
}
void Read_fields() {
    if (!test_switch){
        prepare_input_sources();
        print_stage("Reading input field data from HDF5...");
        Array<int,1> s1,s2, s3;
        if (two_dimension_switch){
            if (scalar_switch) { get_input_shape("", TName, TdName, s1); resize_input(); if (!load_structured_grid_spacing(TName+".h5")) calculate_grid_spacing(); read_2D(T_2D,"", TName, TdName); }
            else { 
                get_input_shape("", UName, UdName, s1); get_input_shape("", WName, WdName, s2);
                if (!compare(s1,s2)) { if (rank_mpi==0) cerr<<"\nIncompatible dimension data\n\n"; h5::finalize(); finalize_mpi_runtime(); exit(1); }
                resize_input(); if (!load_structured_grid_spacing(UName+".h5")) calculate_grid_spacing(); read_2D(V1_2D,"", UName, UdName); read_2D(V3_2D,"", WName, WdName);
            }
        } else {
            if (scalar_switch) { get_input_shape("", TName, TdName, s1); resize_input(); if (!load_structured_grid_spacing(TName+".h5")) calculate_grid_spacing(); read_3D(T, "", TName, TdName); }
            else {
            	get_input_shape("", UName, UdName, s1); get_input_shape("", VName, VdName, s2); get_input_shape("", WName, WdName, s3);
            	if (!compare(s1,s2) || !compare(s2,s3)) { if (rank_mpi==0) cerr<<"\nIncompatible dimension data\n\n"; h5::finalize(); finalize_mpi_runtime(); exit(1); }
            	resize_input(); if (!load_structured_grid_spacing(UName+".h5")) calculate_grid_spacing(); read_3D(V1, "", UName, UdName); read_3D(V2, "", VName, VdName); read_3D(V3, "", WName, WdName);
            }
        }
    } else {
        if (rank_mpi==0) cout<<"\nWARNING: The code is running in TEST mode. Generating fields internally.\n";
        resize_input(); calculate_grid_spacing();
        if (two_dimension_switch) scalar_switch ? Read_Init(T_2D) : Read_Init(V1_2D, V3_2D);
        else scalar_switch ? Read_Init(T) : Read_Init(V1, V2, V3);
    }
    print_stage("Input fields are loaded. Computing global energy diagnostics...");
    ComputeAndPrintTKE();
}

void resize_SFs(){
    if (rank_mpi==0) {
        if (not two_dimension_switch) {
            if (scalar_switch) { SF_Grid_scalar.resize(Nx/2, Ny/2, Nz/2, q2-q1+1); SF_Grid_scalar = 0; }
            else { SF_Grid_pll.resize(Nx/2, Ny/2, Nz/2, q2-q1+1); SF_Grid_pll = 0; if (not longitudinal) { SF_Grid_perp.resize(Nx/2, Ny/2, Nz/2, q2-q1+1); SF_Grid_perp = 0; } }
        } else {
            if (scalar_switch) { SF_Grid2D_scalar.resize(Nx/2, Nz/2, q2-q1+1); SF_Grid2D_scalar = 0; }
            else { SF_Grid2D_pll.resize(Nx/2, Nz/2, q2-q1+1); SF_Grid2D_pll = 0; if (not longitudinal) { SF_Grid2D_perp.resize(Nx/2, Nz/2, q2-q1+1); SF_Grid2D_perp = 0; } }
        }   
    }
}

void calc_SFs() {
    if (two_dimension_switch) scalar_switch ? SF_scalar_2D(T_2D) : (longitudinal ? SFunc_long_2D(V1_2D, V3_2D) : SFunc2D(V1_2D, V3_2D));
    else scalar_switch ? SF_scalar_3D(T) : (longitudinal ? SFunc_long_3D(V1, V2, V3) : SFunc3D(V1, V2, V3));
}

void write_SFs() {
    if (rank_mpi==0){
        ensure_directory_exists(output_dir);
        if (two_dimension_switch) {
            if (scalar_switch) write_3D(SF_Grid2D_scalar, SF_Grid_scalar_name);
            else { write_3D(SF_Grid2D_pll, SF_Grid_pll_name); if (not longitudinal) write_3D(SF_Grid2D_perp, SF_Grid_perp_name); }
        } else {
            if (scalar_switch) write_4D(SF_Grid_scalar, SF_Grid_scalar_name);
            else { write_4D(SF_Grid_pll, SF_Grid_pll_name); if (not longitudinal) write_4D(SF_Grid_perp, SF_Grid_perp_name); }
        }
        cout<<"\nWriting completed\n";
    }
}

void test_cases() {
    if(rank_mpi==0){
        cout<<"\nCOMMENCING TESTING OF THE CODE.\n";
        if (scalar_switch) two_dimension_switch ? SCALAR_TEST_CASE_2D() : SCALAR_TEST_CASE_3D();
        else two_dimension_switch ? VECTOR_TEST_CASE_2D() : VECTOR_TEST_CASE_3D();
    }
}

void get_rank(int rank, int py, int& rankx, int& ranky){ ranky=rank%py; rankx=(rank-ranky)/py; }

void compute_index_list(Array<int,1>& index_list, int Nx, int px, int rank){
    int list_size=Nx/px; index_list.resize(list_size);
    for (int i=0; i<list_size; i+=2){ index_list(i)=rank+i*px; if (px!=Nx) index_list(i+1)=Nx-1-index_list(i); }
}

void compute_index_list(Array<int,3>& index_list, int Nx, int Ny){
    int list_size=(Nx*Ny)/(4*P); index_list.resize(list_size,2,P);
    int py=(P/px); Array<int,1> x, y; int rankx,ranky; int nx=Nx/(2*px),ny=Ny/(2*py);
    for (int rank_id=0; rank_id<P; rank_id++){ get_rank(rank_id, py, rankx, ranky); compute_index_list(x, Nx/2, px, rankx); compute_index_list(y, Ny/2, py, ranky);
        for (int i=0; i<nx; i++){ index_list(Range(ny*i,(i+1)*ny-1),0,rank_id)=x(i); index_list(Range(ny*i,(i+1)*ny-1),1,rank_id)=y(Range::all()); }
    }
}

void VECTOR_TEST_CASE_3D() {	
    double epsilon=1e-10, max=0; Array<double,3> test1,test2;
	if (longitudinal){
		test1.resize(Nx/2,Ny/2,Nz/2);
		for (int order=0 ; order<=q2-q1; order++){
			read_3D(test1,output_dir+"/",SF_Grid_pll_name,SF_Grid_pll_name+int_to_str(order+q1));
			for (int i=0; i<test1.extent(0); i++) for (int j=0; j<test1.extent(1); j++) for (int k=0; k<test1.extent(2); k++){
				double lx=dx*i, ly=dy*j, lz=dz*k; double err = (lx*lx + ly*ly + lz*lz > epsilon) ? abs((test1(i,j,k)-pow(lx*lx+ly*ly+lz*lz,(order+q1)/2.))/pow(lx*lx+ly*ly+lz*lz,(order+q1)/2.)) : abs(test1(i,j,k));
                if (err > max) max = err;
			}
		}
	} else {
        test1.resize(Nx/2,Ny/2,Nz/2); test2.resize(Nx/2,Ny/2,Nz/2);
		for (int order=0 ; order<=q2-q1; order++){
			read_3D(test1,output_dir+"/",SF_Grid_pll_name,SF_Grid_pll_name+int_to_str(order+q1)); read_3D(test2,output_dir+"/",SF_Grid_perp_name,SF_Grid_perp_name+int_to_str(order+q1));
			for (int i=0; i<test1.extent(0); i++) for (int j=0; j<test1.extent(1); j++) for (int k=0; k<test1.extent(2); k++){
				double lx=dx*i, ly=dy*j, lz=dz*k; double err1 = (lx*lx + ly*ly + lz*lz > epsilon) ? abs((test1(i,j,k)-pow(lx*lx+ly*ly+lz*lz,(order+q1)/2.))/pow(lx*lx+ly*ly+lz*lz,(order+q1)/2.)) : abs(test1(i,j,k));
                double err2 = abs(test2(i,j,k)); if (err1 > max) max = err1; if (err2 > max) max = err2;
			}
		}
	}
	if (rank_mpi==0) { max > epsilon ? cout<<"\nVECTOR_3D: TEST_FAILED\n" : cout<<"\nVECTOR_3D: TEST_PASSED\n"; cout<<"MAXIMUM ERROR: "<<max<<endl; }
}

void VECTOR_TEST_CASE_2D() {	
	double epsilon=1e-10, max=0; Array<double,2> test1,test2;
	if (longitudinal){
		test1.resize(Nx/2,Nz/2);
		for (int order=0 ; order<=q2-q1; order++){
			read_2D(test1,output_dir+"/",SF_Grid_pll_name, SF_Grid_pll_name+int_to_str(order+q1));
			for (int i=0; i<test1.extent(0); i++) for (int k=0; k<test1.extent(1); k++){
				double lx=dx*i, lz=dz*k; double err = ((lx*lx + lz*lz)>epsilon) ? abs((test1(i,k)-pow(lx*lx+lz*lz,(order+q1)/2.))/pow(lx*lx+lz*lz,(order+q1)/2.)) : abs(test1(i,k));
                if (err > max) max=err;
			}
		}
	} else {
		test1.resize(Nx/2,Nz/2); test2.resize(Nx/2,Nz/2);
		for (int order=0 ; order<=q2-q1; order++){
			read_2D(test1,output_dir+"/",SF_Grid_pll_name,SF_Grid_pll_name+int_to_str(order+q1)); read_2D(test2,output_dir+"/",SF_Grid_perp_name,SF_Grid_perp_name+int_to_str(order+q1));
			for (int i=0; i<test1.extent(0); i++) for (int k=0; k<test1.extent(1); k++){
				double lx=dx*i, lz=dz*k; double err1 = ((lx*lx + lz*lz)>epsilon) ? abs((test1(i,k)-pow(lx*lx+lz*lz,(order+q1)/2.))/pow(lx*lx+lz*lz,(order+q1)/2.)) : abs(test1(i,k));
                double err2 = abs(test2(i,k)); if (err1 > max) max = err1; if (err2 > max) max = err2;
			}
		}
	}
	if (rank_mpi==0) { max > epsilon ? cout<<"\nVECTOR_2D: TEST_FAILED\n" : cout<<"\nVECTOR_2D: TEST_PASSED\n"; cout<<"MAXIMUM ERROR: "<<max<<endl; }
}

void SCALAR_TEST_CASE_2D() {
    double epsilon=1e-10, max=0; Array<double,2> test1; test1.resize(Nx/2,Nz/2);
	for (int order=0 ; order<=q2-q1; order++){
		read_2D(test1,output_dir+"/",SF_Grid_scalar_name, SF_Grid_scalar_name+int_to_str(order+q1));
		for (int i=0; i<test1.extent(0); i++) for (int k=0; k<test1.extent(1); k++){
			double lx=dx*i, lz=dz*k; double err = (abs(lx+lz)>epsilon) ? abs((test1(i,k)-pow(lx+lz,(order+q1)))/pow(lx+lz,(order+q1))) : abs(test1(i,k));
            if (err>max) max = err;
		}
	}
	if (rank_mpi==0) { max > epsilon ? cout<<"\nSCALAR_2D: TEST_FAILED\n" : cout<<"\nSCALAR_2D: TEST_PASSED\n"; cout<<"MAXIMUM ERROR: "<<max<<endl; }
}

void SCALAR_TEST_CASE_3D(){
	double epsilon=1e-10, max=0; Array<double,3> test1; test1.resize(Nx/2,Ny/2,Nz/2);
	for (int order=0 ; order<=q2-q1; order++){
		read_3D(test1,output_dir+"/",SF_Grid_scalar_name, SF_Grid_scalar_name+int_to_str(order+q1));
		for (int i=0; i<test1.extent(0); i++) for (int j=0; j<test1.extent(1); j++) for (int k=0; k<test1.extent(2); k++){
			double lx=dx*i, ly=dy*j, lz=dz*k; double err = (abs(lx+ly+lz)>epsilon) ? abs((test1(i,j,k)-pow(lx+ly+lz,(order+q1)))/pow(lx+ly+lz,(order+q1))) : abs(test1(i,j,k));
            if (err>max) max=err;
		}
	}
	if (rank_mpi==0) { max > epsilon ? cout<<"\nSCALAR_3D: TEST_FAILED\n" : cout<<"\nSCALAR_3D: TEST_PASSED\n"; cout<<"MAXIMUM ERROR: "<<max<<endl; }
}

string int_to_str(int n) { stringstream ss; ss << n; return ss.str(); }
void compute_time_elapsed(timeval s, timeval e, double& res){ res = ((e.tv_sec-s.tv_sec)*1000000u + e.tv_usec-s.tv_usec)/1.0e6; }

inline double normalized_moment(const Array<double,3>& values, int q_order, int count) {
    if (q_order == 2) return sum(values*values)/count;
    if (q_order == 3) return sum(values*values*values)/count;
    return sum(pow(values, q_order))/count;
}

inline double normalized_moment(const Array<double,2>& values, int q_order, int count) {
    if (q_order == 2) return sum(values*values)/count;
    if (q_order == 3) return sum(values*values*values)/count;
    return sum(pow(values, q_order))/count;
}

void write_4D(Array<double,4> A, string file) {
  int nx=A.extent(0), ny=A.extent(1), nz=A.extent(2); h5::File f(output_file_path(file), "w");
  for (int q=q1; q<=q2; q++) {
      if (rank_mpi==0) cout<<"Writing "<<q<<" order to file.\n"; string qstr = int_to_str(q);
      h5::Dataset ds = f.create_dataset(file+qstr, h5::shape(nx,ny,nz), "double");
      Array<double,3> t(nx,ny,nz); t = A(Range::all(),Range::all(),Range::all(),q-q1);
      ds << t.data();
  }
}

void write_3D(Array<double,3> A, string file) {
  int nx=A.extent(0), nz=A.extent(1); h5::File f(output_file_path(file), "w");
  for (int q=q1; q<=q2; q++) {
      if (rank_mpi==0) cout<<"Writing "<<q<<" order to file.\n"; string qstr = int_to_str(q);
      h5::Dataset ds = f.create_dataset(file+qstr, h5::shape(nx,nz), "double");
      Array<double,2> t(nx,nz); t = A(Range::all(),Range::all(),q-q1);
      ds << t.data();
  }
}

void show_checklist(){ if (rank_mpi==0) cerr<<"Error: Check inputs in 'in/' folder.\n"; }
void read_2D(Array<double,2> A, string fold, string file, string dset) {
    string path = fold+file+".h5";
    hid_t file_id = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t dset_id = H5Dopen2(file_id, dset.c_str(), H5P_DEFAULT);
    hid_t space_id = H5Dget_space(dset_id);
    int rank = H5Sget_simple_extent_ndims(space_id);
    std::vector<hsize_t> dims(rank, 1);
    H5Sget_simple_extent_dims(space_id, &dims[0], NULL);
    if (rank == 2 && dims[0] == static_cast<hsize_t>(A.extent(0)) && dims[1] == static_cast<hsize_t>(A.extent(1))) {
        H5Dread(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, A.data());
    } else if (rank == 3 && dims[0] >= static_cast<hsize_t>(A.extent(0)) && dims[2] >= static_cast<hsize_t>(A.extent(1))) {
        hsize_t start[3] = {0, 0, 0};
        hsize_t count[3] = {static_cast<hsize_t>(A.extent(0)), 1, static_cast<hsize_t>(A.extent(1))};
        H5Sselect_hyperslab(space_id, H5S_SELECT_SET, start, NULL, count, NULL);
        hid_t mem_space = H5Screate_simple(3, count, NULL);
        std::vector<double> tmp(static_cast<std::size_t>(A.extent(0)) * A.extent(1), 0.0);
        H5Dread(dset_id, H5T_NATIVE_DOUBLE, mem_space, space_id, H5P_DEFAULT, &tmp[0]);
        for (int i=0; i<A.extent(0); ++i) for (int k=0; k<A.extent(1); ++k) A(i,k) = tmp[i*A.extent(1) + k];
        H5Sclose(mem_space);
    } else {
        H5Sclose(space_id);
        H5Dclose(dset_id);
        H5Fclose(file_id);
        throw std::runtime_error("Unsupported dataset rank for 2D read");
    }
    H5Sclose(space_id);
    H5Dclose(dset_id);
    H5Fclose(file_id);
}
void read_3D(Array<double,3> A, string fold, string file, string dset) {
    string path = fold+file+".h5";
    hid_t file_id = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t dset_id = H5Dopen2(file_id, dset.c_str(), H5P_DEFAULT);
    hid_t space_id = H5Dget_space(dset_id);
    int rank = H5Sget_simple_extent_ndims(space_id);
    std::vector<hsize_t> dims(rank, 1);
    H5Sget_simple_extent_dims(space_id, &dims[0], NULL);
    hsize_t count[3] = {static_cast<hsize_t>(A.extent(0)), static_cast<hsize_t>(A.extent(1)), static_cast<hsize_t>(A.extent(2))};
    if (rank == 3 && dims[0] == count[0] && dims[1] == count[1] && dims[2] == count[2]) {
        H5Dread(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, A.data());
    } else if (rank == 3 && dims[0] >= count[0] && dims[1] >= count[1] && dims[2] >= count[2]) {
        hsize_t start[3] = {0, 0, 0};
        H5Sselect_hyperslab(space_id, H5S_SELECT_SET, start, NULL, count, NULL);
        hid_t mem_space = H5Screate_simple(3, count, NULL);
        H5Dread(dset_id, H5T_NATIVE_DOUBLE, mem_space, space_id, H5P_DEFAULT, A.data());
        H5Sclose(mem_space);
    } else {
        H5Sclose(space_id);
        H5Dclose(dset_id);
        H5Fclose(file_id);
        throw std::runtime_error("Unsupported dataset rank for 3D read");
    }
    H5Sclose(space_id);
    H5Dclose(dset_id);
    H5Fclose(file_id);
}

void Read_Init(Array<double,3>& Ux, Array<double,3>& Uy, Array<double,3>& Uz){
  if (rank_mpi==0) cout<<"\nGenerating 3D velocity field: U = [x, y, z] \n";
  for (int i=0; i<Nx; i++) for (int j=0; j<Ny; j++) for (int k=0; k<Nz; k++) { Ux(i, j, k) = i*dx; Uy(i, j, k) = j*dy; Uz(i, j, k) = k*dz; }
}
void Read_Init(Array<double,2>& Ux, Array<double,2>& Uz){
	if (rank_mpi==0) cout<<"\nGenerating 2D velocity field: U = [x, z] \n";
    for (int i=0;i<Nx;i++) for (int k=0;k<Nz;k++){ Ux(i, k) = i*dx; Uz(i, k) = k*dz; }
}
void Read_Init(Array<double,2>& T) {
	if (rank_mpi==0) cout<<"\nGenerating scalar field: T = x + z \n";
    for (int i=0;i<Nx;i++) for (int k=0;k<Nz;k++) T(i, k) = i*dx + k*dz;
}
void Read_Init(Array<double,3>& T) {
	if (rank_mpi==0) cout<<"\nGenerating scalar field: T = x + y + z \n";
    for (int i=0;i<Nx;i++) for (int j=0;j<Ny;j++) for (int k=0;k<Nz;k++) T(i, j, k) = i*dx + j*dy + k*dz;
}

void SFunc3D(const Array<double,3>& Ux, const Array<double,3>& Uy, const Array<double,3>& Uz) {
	if (rank_mpi==0) cout<<"\nComputing longitudinal and transverse S(lx, ly, lz) using 3D velocity field data..\n";
    int c_per_proc = Nx*Ny/(4*P); Array<int, 3> idx; compute_index_list(idx, Nx, Ny);
    int total_offsets = c_per_proc * (Nz/2);
    int order_count = q2-q1+1;
    ProgressState progress = make_progress_state();
    std::vector<double> local_values(2*order_count, 0.0);
    std::vector<double> gathered_values;
    // Historical behavior allocated shape-specific work arrays inside the lag loops below.
    // Reuse max-sized buffers instead and operate only on the active subranges for each lag.
    Array<double,3> dUx(Nx,Ny,Nz), dUy(Nx,Ny,Nz), dUz(Nx,Ny,Nz), dUpll(Nx,Ny,Nz);
    if (rank_mpi == 0) gathered_values.resize(2*order_count*P, 0.0);
    if (rank_mpi == 0) cout << "[fastSF] 3D velocity workload: " << total_offsets << " displacement offsets across q=" << q1 << ".." << q2 << endl;
    if (c_per_proc > 0) print_progress_start("3D velocity SF", total_offsets, idx(0, 0, rank_mpi), idx(0, 1, rank_mpi), 0);
    for (int ix=0; ix<c_per_proc; ix++){
        int x=idx(ix, 0, rank_mpi), y=idx(ix, 1, rank_mpi);
  		for(int z=0; z<Nz/2; z++){
            bool first_offset = (ix == 0 && z == 0);
            print_offset_stage("3D velocity SF", first_offset ? "starting first lag-offset evaluation" : "still sweeping lag offsets", progress, x, y, z, first_offset);
            if (x == 0 && y == 0 && z == 0) {
                std::fill(local_values.begin(), local_values.end(), 0.0);
                // Historical behavior evaluated zero lag through the full temporary-array path below and
                // then overwrote the stored origin with zero. Keep the old nonzero-lag path intact, but
                // bypass the redundant local math here while preserving the root-visible output indices.
                print_offset_stage("3D velocity SF", "zero lag detected; bypassing temporary-array reductions and sending exact zeros", progress, x, y, z, true);
                print_offset_stage("3D velocity SF", "gathering all requested q-orders for zero lag in one MPI_Gather", progress, x, y, z, true);
                MPI_Gather(local_values.data(), static_cast<int>(local_values.size()), MPI_DOUBLE,
                           rank_mpi == 0 ? gathered_values.data() : NULL,
                           static_cast<int>(local_values.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
                if (rank_mpi==0) {
                    for (int rank_id=0; rank_id<P; rank_id++) {
                        int x_rank = idx(ix, 0, rank_id), y_rank = idx(ix, 1, rank_id);
                        for (int p=0; p<order_count; p++) {
                            int base = rank_id*(2*order_count);
                            SF_Grid_pll(x_rank,y_rank,z,p)=gathered_values[base+p];
                            SF_Grid_perp(x_rank,y_rank,z,p)=gathered_values[base+order_count+p];
                        }
                    }
                }
                print_loop_progress("3D velocity SF", ix*(Nz/2) + z + 1, total_offsets, progress, x, y, z);
                continue;
            }
        	int cnt=(Nx-x)*(Ny-y)*(Nz-z); double lx=x*dx, ly=y*dy, lz=z*dz, r=sqrt(lx*lx+ly*ly+lz*lz); if (r<1e-15) r=1e-15;
            Range rx(0,Nx-x-1), ry(0,Ny-y-1), rz(0,Nz-z-1);
        	dUx(rx,ry,rz) = Ux(Range(x,Nx-1),Range(y,Ny-1),Range(z,Nz-1)) - Ux(Range(0,Nx-x-1),Range(0,Ny-y-1),Range(0,Nz-z-1));
        	dUy(rx,ry,rz) = Uy(Range(x,Nx-1),Range(y,Ny-1),Range(z,Nz-1)) - Uy(Range(0,Nx-x-1),Range(0,Ny-y-1),Range(0,Nz-z-1));
        	dUz(rx,ry,rz) = Uz(Range(x,Nx-1),Range(y,Ny-1),Range(z,Nz-1)) - Uz(Range(0,Nx-x-1),Range(0,Ny-y-1),Range(0,Nz-z-1));
            print_offset_stage("3D velocity SF", "computed velocity differences; projecting longitudinal/transverse components", progress, x, y, z, first_offset);
            dUpll(rx,ry,rz) = (lx*dUx(rx,ry,rz)+ly*dUy(rx,ry,rz)+lz*dUz(rx,ry,rz))/r;
            dUx(rx,ry,rz) = dUx(rx,ry,rz)-dUpll(rx,ry,rz)*lx/r;
            dUy(rx,ry,rz) = dUy(rx,ry,rz)-dUpll(rx,ry,rz)*ly/r;
            dUz(rx,ry,rz) = dUz(rx,ry,rz)-dUpll(rx,ry,rz)*lz/r;
            dUx(rx,ry,rz) = sqrt(dUx(rx,ry,rz)*dUx(rx,ry,rz)+dUy(rx,ry,rz)*dUy(rx,ry,rz)+dUz(rx,ry,rz)*dUz(rx,ry,rz));
            print_offset_stage("3D velocity SF", "reducing requested q-orders and gathering rank results", progress, x, y, z, first_offset);
        	for (int p=0; p<=q2-q1; p++){
                int q_order = q1 + p;
                print_q_stage("3D velocity SF", "starting local reduction for order", progress, x, y, z, q_order, first_offset);
        		local_values[p] = normalized_moment(dUpll(rx,ry,rz), q_order, cnt);
                local_values[order_count+p] = normalized_moment(dUx(rx,ry,rz), q_order, cnt);
        	}
            print_offset_stage("3D velocity SF", "completed local reductions; gathering all requested q-orders for this lag offset", progress, x, y, z, first_offset);
            MPI_Gather(local_values.data(), static_cast<int>(local_values.size()), MPI_DOUBLE,
                       rank_mpi == 0 ? gathered_values.data() : NULL,
                       static_cast<int>(local_values.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
            if (rank_mpi==0) {
                for (int rank_id=0; rank_id<P; rank_id++) {
                    int x_rank = idx(ix, 0, rank_id), y_rank = idx(ix, 1, rank_id);
                    int base = rank_id*(2*order_count);
                    for (int p=0; p<order_count; p++) {
                        SF_Grid_pll(x_rank,y_rank,z,p)=gathered_values[base+p];
                        SF_Grid_perp(x_rank,y_rank,z,p)=gathered_values[base+order_count+p];
                    }
                }
            }
            print_loop_progress("3D velocity SF", ix*(Nz/2) + z + 1, total_offsets, progress, x, y, z);
  		}
  	}
    if (rank_mpi==0) { SF_Grid_pll(0,0,0,Range::all())=0; SF_Grid_perp(0,0,0,Range::all())=0; }
}

void SFunc_long_3D(const Array<double,3>& Ux, const Array<double,3>& Uy, const Array<double,3>& Uz) {
if (rank_mpi==0) cout<<"\nComputing longitudinal S(lx, ly, lz) using 3D velocity field data..\n";
    int c_per_proc = Nx*Ny/(4*P); Array<int, 3> idx; compute_index_list(idx, Nx, Ny);
    int total_offsets = c_per_proc * (Nz/2);
    int order_count = q2-q1+1;
    ProgressState progress = make_progress_state();
    std::vector<double> local_values(order_count, 0.0);
    std::vector<double> gathered_values;
    Array<double,3> dUx(Nx,Ny,Nz), dUy(Nx,Ny,Nz), dUz(Nx,Ny,Nz), dUpll(Nx,Ny,Nz);
    if (rank_mpi == 0) gathered_values.resize(order_count*P, 0.0);
    if (rank_mpi == 0) cout << "[fastSF] 3D longitudinal workload: " << total_offsets << " displacement offsets across q=" << q1 << ".." << q2 << endl;
    if (c_per_proc > 0) print_progress_start("3D longitudinal SF", total_offsets, idx(0, 0, rank_mpi), idx(0, 1, rank_mpi), 0);
    for (int ix=0; ix<c_per_proc; ix++){
        int x=idx(ix, 0, rank_mpi), y=idx(ix, 1, rank_mpi);
  		for(int z=0; z<Nz/2; z++){
            bool first_offset = (ix == 0 && z == 0);
            print_offset_stage("3D longitudinal SF", first_offset ? "starting first lag-offset evaluation" : "still sweeping lag offsets", progress, x, y, z, first_offset);
            if (x == 0 && y == 0 && z == 0) {
                std::fill(local_values.begin(), local_values.end(), 0.0);
                // Historical behavior evaluated zero lag through the full temporary-array path below and
                // then overwrote the stored origin with zero. Keep the old nonzero-lag path intact, but
                // bypass the redundant local math here while preserving the root-visible output indices.
                print_offset_stage("3D longitudinal SF", "zero lag detected; bypassing temporary-array reductions and sending exact zeros", progress, x, y, z, true);
                print_offset_stage("3D longitudinal SF", "gathering all requested q-orders for zero lag in one MPI_Gather", progress, x, y, z, true);
                MPI_Gather(local_values.data(), static_cast<int>(local_values.size()), MPI_DOUBLE,
                           rank_mpi == 0 ? gathered_values.data() : NULL,
                           static_cast<int>(local_values.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
                if (rank_mpi==0) {
                    for (int rank_id=0; rank_id<P; rank_id++) {
                        int x_rank = idx(ix, 0, rank_id), y_rank = idx(ix, 1, rank_id);
                        for (int p=0; p<order_count; p++) SF_Grid_pll(x_rank,y_rank,z,p)=gathered_values[rank_id*order_count+p];
                    }
                }
                print_loop_progress("3D longitudinal SF", ix*(Nz/2) + z + 1, total_offsets, progress, x, y, z);
                continue;
            }
    		int cnt=(Nx-x)*(Ny-y)*(Nz-z); double lx=x*dx, ly=y*dy, lz=z*dz, r=sqrt(lx*lx+ly*ly+lz*lz); if (r<1e-15) r=1e-15;
            Range rx(0,Nx-x-1), ry(0,Ny-y-1), rz(0,Nz-z-1);
    		dUx(rx,ry,rz) = Ux(Range(x,Nx-1),Range(y,Ny-1),Range(z,Nz-1)) - Ux(Range(0,Nx-x-1),Range(0,Ny-y-1),Range(0,Nz-z-1));
    		dUy(rx,ry,rz) = Uy(Range(x,Nx-1),Range(y,Ny-1),Range(z,Nz-1)) - Uy(Range(0,Nx-x-1),Range(0,Ny-y-1),Range(0,Nz-z-1));
    		dUz(rx,ry,rz) = Uz(Range(x,Nx-1),Range(y,Ny-1),Range(z,Nz-1)) - Uz(Range(0,Nx-x-1),Range(0,Ny-y-1),Range(0,Nz-z-1));
            print_offset_stage("3D longitudinal SF", "computed velocity differences; projecting longitudinal component", progress, x, y, z, first_offset);
            dUpll(rx,ry,rz) = (lx*dUx(rx,ry,rz)+ly*dUy(rx,ry,rz)+lz*dUz(rx,ry,rz))/r;
            print_offset_stage("3D longitudinal SF", "reducing requested q-orders and gathering rank results", progress, x, y, z, first_offset);
    		for (int p=0; p<=q2-q1; p++){
                int q_order = q1 + p;
                print_q_stage("3D longitudinal SF", "starting local reduction for order", progress, x, y, z, q_order, first_offset);
    			local_values[p] = normalized_moment(dUpll(rx,ry,rz), q_order, cnt);
    		}
            print_offset_stage("3D longitudinal SF", "completed local reductions; gathering all requested q-orders for this lag offset", progress, x, y, z, first_offset);
            MPI_Gather(local_values.data(), static_cast<int>(local_values.size()), MPI_DOUBLE,
                       rank_mpi == 0 ? gathered_values.data() : NULL,
                       static_cast<int>(local_values.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
            if (rank_mpi==0) {
                for (int rank_id=0; rank_id<P; rank_id++) {
                    int x_rank = idx(ix, 0, rank_id), y_rank = idx(ix, 1, rank_id);
                    for (int p=0; p<order_count; p++) SF_Grid_pll(x_rank,y_rank,z,p)=gathered_values[rank_id*order_count+p];
                }
            }
            print_loop_progress("3D longitudinal SF", ix*(Nz/2) + z + 1, total_offsets, progress, x, y, z);
  		}
  	}
    if (rank_mpi==0) SF_Grid_pll(0,0,0,Range::all())=0;
}

void SFunc2D(const Array<double,2>& Ux, const Array<double,2>& Uz) {
     if (rank_mpi==0) cout<<"\nComputing longitudinal and transverse S(lx, lz) using 2D velocity field data..\n";
    int p_per_proc = Nx*Nz/(4*P); Array<int, 3> idx; compute_index_list(idx, Nx, Nz);
    int order_count = q2-q1+1;
    ProgressState progress = make_progress_state();
    std::vector<double> local_values(2*order_count, 0.0);
    std::vector<double> gathered_values;
    Array<double,2> dUx(Nx,Nz), dUz(Nx,Nz), dUpll(Nx,Nz);
    if (rank_mpi == 0) gathered_values.resize(2*order_count*P, 0.0);
    if (rank_mpi == 0) cout << "[fastSF] 2D velocity workload: " << p_per_proc << " displacement offsets across q=" << q1 << ".." << q2 << endl;
    if (p_per_proc > 0) print_progress_start("2D velocity SF", p_per_proc, idx(0, 0, rank_mpi), -1, idx(0, 1, rank_mpi));
    for (int ix=0; ix<p_per_proc; ix++){
        int x=idx(ix, 0, rank_mpi), z=idx(ix, 1, rank_mpi);
        if (x == 0 && z == 0) {
            std::fill(local_values.begin(), local_values.end(), 0.0);
            // Historical behavior evaluated zero lag through the full temporary-array path below and
            // then overwrote the stored origin with zero. Keep the old nonzero-lag path intact, but
            // bypass the redundant local math here while preserving the root-visible output indices.
            MPI_Gather(local_values.data(), static_cast<int>(local_values.size()), MPI_DOUBLE,
                       rank_mpi == 0 ? gathered_values.data() : NULL,
                       static_cast<int>(local_values.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
            if (rank_mpi==0) {
                for (int rank_id=0; rank_id<P; rank_id++) {
                    int x_rank = idx(ix, 0, rank_id), z_rank = idx(ix, 1, rank_id);
                    int base = rank_id*(2*order_count);
                    for (int p=0; p<order_count; p++) {
                        SF_Grid2D_pll(x_rank,z_rank,p)=gathered_values[base+p];
                        SF_Grid2D_perp(x_rank,z_rank,p)=gathered_values[base+order_count+p];
                    }
                }
            }
            print_loop_progress("2D velocity SF", ix + 1, p_per_proc, progress, x, -1, z);
            continue;
        }
        int cnt=(Nx-x)*(Nz-z); double lx=x*dx, lz=z*dz, r=sqrt(lx*lx+lz*lz); if (r<1e-15) r=1e-15;
        Range rx(0,Nx-x-1), rz(0,Nz-z-1);
        dUx(rx,rz) = Ux(Range(x,Nx-1),Range(z,Nz-1)) - Ux(Range(0,Nx-x-1),Range(0,Nz-z-1));
        dUz(rx,rz) = Uz(Range(x,Nx-1),Range(z,Nz-1)) - Uz(Range(0,Nx-x-1),Range(0,Nz-z-1));
        dUpll(rx,rz) = (lx*dUx(rx,rz)+lz*dUz(rx,rz))/r;
        dUx(rx,rz) = dUx(rx,rz)-dUpll(rx,rz)*lx/r;
        dUz(rx,rz) = dUz(rx,rz)-dUpll(rx,rz)*lz/r;
        dUx(rx,rz) = sqrt(dUx(rx,rz)*dUx(rx,rz)+dUz(rx,rz)*dUz(rx,rz));
    	for (int p=0; p<=q2-q1; p++){
            int q_order = q1 + p;
            local_values[p] = normalized_moment(dUpll(rx,rz), q_order, cnt);
            local_values[order_count+p] = normalized_moment(dUx(rx,rz), q_order, cnt);
        } 
        MPI_Gather(local_values.data(), static_cast<int>(local_values.size()), MPI_DOUBLE,
                   rank_mpi == 0 ? gathered_values.data() : NULL,
                   static_cast<int>(local_values.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank_mpi==0) {
            for (int rank_id=0; rank_id<P; rank_id++) {
                int x_rank = idx(ix, 0, rank_id), z_rank = idx(ix, 1, rank_id);
                int base = rank_id*(2*order_count);
                for (int p=0; p<order_count; p++) {
                    SF_Grid2D_pll(x_rank,z_rank,p)=gathered_values[base+p];
                    SF_Grid2D_perp(x_rank,z_rank,p)=gathered_values[base+order_count+p];
                }
            }
        }
        print_loop_progress("2D velocity SF", ix + 1, p_per_proc, progress, x, -1, z);
    }
    if (rank_mpi==0) { SF_Grid2D_pll(0,0,Range::all())=0; SF_Grid2D_perp(0,0,Range::all())=0; }
}

void SFunc_long_2D(const Array<double,2>& Ux, const Array<double,2>& Uz) {
     if (rank_mpi==0) cout<<"\nComputing longitudinal S(lx, lz) using 2D velocity field data..\n";
    int p_per_proc = Nx*Nz/(4*P); Array<int, 3> idx; compute_index_list(idx, Nx, Nz);
    int order_count = q2-q1+1;
    ProgressState progress = make_progress_state();
    std::vector<double> local_values(order_count, 0.0);
    std::vector<double> gathered_values;
    Array<double,2> dUx(Nx,Nz), dUz(Nx,Nz), dUpll(Nx,Nz);
    if (rank_mpi == 0) gathered_values.resize(order_count*P, 0.0);
    if (rank_mpi == 0) cout << "[fastSF] 2D longitudinal workload: " << p_per_proc << " displacement offsets across q=" << q1 << ".." << q2 << endl;
    if (p_per_proc > 0) print_progress_start("2D longitudinal SF", p_per_proc, idx(0, 0, rank_mpi), -1, idx(0, 1, rank_mpi));
    for (int ix=0; ix<p_per_proc; ix++){
        int x=idx(ix, 0, rank_mpi), z=idx(ix, 1, rank_mpi);
        if (x == 0 && z == 0) {
            std::fill(local_values.begin(), local_values.end(), 0.0);
            // Historical behavior evaluated zero lag through the full temporary-array path below and
            // then overwrote the stored origin with zero. Keep the old nonzero-lag path intact, but
            // bypass the redundant local math here while preserving the root-visible output indices.
            MPI_Gather(local_values.data(), static_cast<int>(local_values.size()), MPI_DOUBLE,
                       rank_mpi == 0 ? gathered_values.data() : NULL,
                       static_cast<int>(local_values.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
            if (rank_mpi==0) {
                for (int rank_id=0; rank_id<P; rank_id++) {
                    int x_rank = idx(ix, 0, rank_id), z_rank = idx(ix, 1, rank_id);
                    for (int p=0; p<order_count; p++) SF_Grid2D_pll(x_rank,z_rank,p)=gathered_values[rank_id*order_count+p];
                }
            }
            print_loop_progress("2D longitudinal SF", ix + 1, p_per_proc, progress, x, -1, z);
            continue;
        }
        int cnt=(Nx-x)*(Nz-z); double lx=x*dx, lz=z*dz, r=sqrt(lx*lx+lz*lz); if (r<1e-15) r=1e-15;
        Range rx(0,Nx-x-1), rz(0,Nz-z-1);
        dUx(rx,rz) = Ux(Range(x,Nx-1),Range(z,Nz-1)) - Ux(Range(0,Nx-x-1),Range(0,Nz-z-1));
        dUz(rx,rz) = Uz(Range(x,Nx-1),Range(z,Nz-1)) - Uz(Range(0,Nx-x-1),Range(0,Nz-z-1));
        dUpll(rx,rz) = (lx*dUx(rx,rz)+lz*dUz(rx,rz))/r;
        for (int p=0; p<=q2-q1; p++){
            int q_order = q1 + p;
            local_values[p] = normalized_moment(dUpll(rx,rz), q_order, cnt);
        }
        MPI_Gather(local_values.data(), static_cast<int>(local_values.size()), MPI_DOUBLE,
                   rank_mpi == 0 ? gathered_values.data() : NULL,
                   static_cast<int>(local_values.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank_mpi==0) {
            for (int rank_id=0; rank_id<P; rank_id++) {
                int x_rank = idx(ix, 0, rank_id), z_rank = idx(ix, 1, rank_id);
                for (int p=0; p<order_count; p++) SF_Grid2D_pll(x_rank,z_rank,p)=gathered_values[rank_id*order_count+p];
            }
        }
        print_loop_progress("2D longitudinal SF", ix + 1, p_per_proc, progress, x, -1, z);
    }
    if (rank_mpi==0) SF_Grid2D_pll(0,0,Range::all())=0;
}

void SF_scalar_3D(const Array<double,3>& T) {
     if (rank_mpi==0) cout<<"\nComputing S(lx, ly, lz) using 3D scalar field data..\n";
    int c_per_proc = Nx*Ny/(4*P); Array<int, 3> idx; compute_index_list(idx, Nx, Ny);
    int total_offsets = c_per_proc * (Nz/2);
    int order_count = q2-q1+1;
    ProgressState progress = make_progress_state();
    std::vector<double> local_values(order_count, 0.0);
    std::vector<double> gathered_values;
    Array<double,3> dT(Nx,Ny,Nz);
    if (rank_mpi == 0) gathered_values.resize(order_count*P, 0.0);
    if (rank_mpi == 0) cout << "[fastSF] 3D scalar workload: " << total_offsets << " displacement offsets across q=" << q1 << ".." << q2 << endl;
    if (c_per_proc > 0) print_progress_start("3D scalar SF", total_offsets, idx(0, 0, rank_mpi), idx(0, 1, rank_mpi), 0);
    for (int ix=0; ix<c_per_proc; ix++){
        int x=idx(ix, 0, rank_mpi), y=idx(ix, 1, rank_mpi);
        for(int z=0; z<Nz/2; z++){
            if (x == 0 && y == 0 && z == 0) {
                std::fill(local_values.begin(), local_values.end(), 0.0);
                // Historical behavior evaluated zero lag through the full temporary-array path below and
                // then overwrote the stored origin with zero. Keep the old nonzero-lag path intact, but
                // bypass the redundant local math here while preserving the root-visible output indices.
                MPI_Gather(local_values.data(), static_cast<int>(local_values.size()), MPI_DOUBLE,
                           rank_mpi == 0 ? gathered_values.data() : NULL,
                           static_cast<int>(local_values.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
                if (rank_mpi==0) {
                    for (int rank_id=0; rank_id<P; rank_id++) {
                        int x_rank = idx(ix, 0, rank_id), y_rank = idx(ix, 1, rank_id);
                        for (int p=0; p<order_count; p++) SF_Grid_scalar(x_rank,y_rank,z,p)=gathered_values[rank_id*order_count+p];
                    }
                }
                print_loop_progress("3D scalar SF", ix*(Nz/2) + z + 1, total_offsets, progress, x, y, z);
                continue;
            }
            int cnt=(Nx-x)*(Ny-y)*(Nz-z); Range rx(0,Nx-x-1), ry(0,Ny-y-1), rz(0,Nz-z-1);
            dT(rx,ry,rz) = T(Range(x,Nx-1),Range(y,Ny-1),Range(z,Nz-1)) - T(Range(0,Nx-x-1),Range(0,Ny-y-1),Range(0,Nz-z-1));
            for (int p=0; p<=q2-q1; p++){
                int q_order = q1 + p;
                local_values[p] = normalized_moment(dT(rx,ry,rz), q_order, cnt);
            }
            MPI_Gather(local_values.data(), static_cast<int>(local_values.size()), MPI_DOUBLE,
                       rank_mpi == 0 ? gathered_values.data() : NULL,
                       static_cast<int>(local_values.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
            if (rank_mpi==0) {
                for (int rank_id=0; rank_id<P; rank_id++) {
                    int x_rank = idx(ix, 0, rank_id), y_rank = idx(ix, 1, rank_id);
                    for (int p=0; p<order_count; p++) SF_Grid_scalar(x_rank,y_rank,z,p)=gathered_values[rank_id*order_count+p];
                }
            }
            print_loop_progress("3D scalar SF", ix*(Nz/2) + z + 1, total_offsets, progress, x, y, z);
        }
    }
    if (rank_mpi==0) SF_Grid_scalar(0,0,0,Range::all())=0;
}

void SF_scalar_2D(const Array<double,2>& T) {
     if (rank_mpi==0) cout<<"\nComputing S(lx, lz) using 2D scalar field data..\n";
    int p_per_proc = Nx*Nz/(4*P); Array<int, 3> idx; compute_index_list(idx, Nx, Nz);
    int order_count = q2-q1+1;
    ProgressState progress = make_progress_state();
    std::vector<double> local_values(order_count, 0.0);
    std::vector<double> gathered_values;
    Array<double,2> dT(Nx,Nz);
    if (rank_mpi == 0) gathered_values.resize(order_count*P, 0.0);
    if (rank_mpi == 0) cout << "[fastSF] 2D scalar workload: " << p_per_proc << " displacement offsets across q=" << q1 << ".." << q2 << endl;
    if (p_per_proc > 0) print_progress_start("2D scalar SF", p_per_proc, idx(0, 0, rank_mpi), -1, idx(0, 1, rank_mpi));
    for (int ix=0; ix<p_per_proc; ix++){
        int x=idx(ix, 0, rank_mpi), z=idx(ix, 1, rank_mpi);
        if (x == 0 && z == 0) {
            std::fill(local_values.begin(), local_values.end(), 0.0);
            // Historical behavior evaluated zero lag through the full temporary-array path below and
            // then overwrote the stored origin with zero. Keep the old nonzero-lag path intact, but
            // bypass the redundant local math here while preserving the root-visible output indices.
            MPI_Gather(local_values.data(), static_cast<int>(local_values.size()), MPI_DOUBLE,
                       rank_mpi == 0 ? gathered_values.data() : NULL,
                       static_cast<int>(local_values.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
            if (rank_mpi==0) {
                for (int rank_id=0; rank_id<P; rank_id++) {
                    int x_rank = idx(ix, 0, rank_id), z_rank = idx(ix, 1, rank_id);
                    for (int p=0; p<order_count; p++) SF_Grid2D_scalar(x_rank,z_rank,p)=gathered_values[rank_id*order_count+p];
                }
            }
            print_loop_progress("2D scalar SF", ix + 1, p_per_proc, progress, x, -1, z);
            continue;
        }
        int cnt=(Nx-x)*(Nz-z); Range rx(0,Nx-x-1), rz(0,Nz-z-1);
        dT(rx,rz) = T(Range(x,Nx-1),Range(z,Nz-1)) - T(Range(0,Nx-x-1),Range(0,Nz-z-1));
        for (int p=0; p<=q2-q1; p++){
            int q_order = q1 + p;
            local_values[p] = normalized_moment(dT(rx,rz), q_order, cnt);
        }
        MPI_Gather(local_values.data(), static_cast<int>(local_values.size()), MPI_DOUBLE,
                   rank_mpi == 0 ? gathered_values.data() : NULL,
                   static_cast<int>(local_values.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank_mpi==0) {
            for (int rank_id=0; rank_id<P; rank_id++) {
                int x_rank = idx(ix, 0, rank_id), z_rank = idx(ix, 1, rank_id);
                for (int p=0; p<order_count; p++) SF_Grid2D_scalar(x_rank,z_rank,p)=gathered_values[rank_id*order_count+p];
            }
        }
        print_loop_progress("2D scalar SF", ix + 1, p_per_proc, progress, x, -1, z);
    }
    if (rank_mpi==0) SF_Grid2D_scalar(0,0,Range::all())=0;
}

void ComputeAndPrintTKE() {
    double local_tke = 0;
    if (two_dimension_switch) { if (scalar_switch) local_tke = sum(pow2(T_2D)); else local_tke = 0.5 * (sum(pow2(V1_2D)) + sum(pow2(V3_2D))); }
    else { if (scalar_switch) local_tke = sum(pow2(T)); else local_tke = 0.5 * (sum(pow2(V1)) + sum(pow2(V2)) + sum(pow2(V3))); }
    double total_tke = 0; MPI_Reduce(&local_tke, &total_tke, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    double total_points = two_dimension_switch ? static_cast<double>(Nx) * Nz : static_cast<double>(Nx) * Ny * Nz;
    if (rank_mpi == 0) cout << "MEAN KINETIC ENERGY (or Scalar MeanSq): " << total_tke / (double(P) * total_points) << endl;
}

void DumpFieldsToTxt() {
    if (rank_mpi != 0) return;
    if (two_dimension_switch) {
        if (scalar_switch) { ofstream f("in/" + TName + ".txt"); f << "# x y z T\n# 2\n# 3\n# 4\n# 5\n# 6\n# 7\n# 8\n"; for (int i=0; i<Nx; i++) for (int k=0; k<Nz; k++) f << i*dx << " " << 0 << " " << k*dz << " " << T_2D(i,k) << "\n"; }
        else { ofstream fu("in/" + UName + ".txt"), fw("in/" + WName + ".txt"); fu << "# x y z vx vy vz\n# 2\n# 3\n# 4\n# 5\n# 6\n# 7\n# 8\n"; fw << "# x y z vx vy vz\n# 2\n# 3\n# 4\n# 5\n# 6\n# 7\n# 8\n"; for (int i=0; i<Nx; i++) for (int k=0; k<Nz; k++) { fu << i*dx << " " << 0 << " " << k*dz << " " << V1_2D(i,k) << " 0 0\n"; fw << i*dx << " " << 0 << " " << k*dz << " " << V3_2D(i,k) << " 0 0\n"; } }
    } else {
        if (scalar_switch) { ofstream f("in/" + TName + ".txt"); f << "# x y z T\n# 2\n# 3\n# 4\n# 5\n# 6\n# 7\n# 8\n"; for (int i=0; i<Nx; i++) for (int j=0; j<Ny; j++) for (int k=0; k<Nz; k++) f << i*dx << " " << j*dy << " " << k*dz << " " << T(i,j,k) << "\n"; }
        else { ofstream fu("in/" + UName + ".txt"), fv("in/" + VName + ".txt"), fw("in/" + WName + ".txt"); fu << "# x y z vx vy vz\n# 2\n# 3\n# 4\n# 5\n# 6\n# 7\n# 8\n"; fv << "# x y z vx vy vz\n# 2\n# 3\n# 4\n# 5\n# 6\n# 7\n# 8\n"; fw << "# x y z vx vy vz\n# 2\n# 3\n# 4\n# 5\n# 6\n# 7\n# 8\n"; for (int i=0; i<Nx; i++) for (int j=0; j<Ny; j++) for (int k=0; k<Nz; k++) { fu << i*dx << " " << j*dy << " " << k*dz << " " << V1(i,j,k) << " 0 0\n"; fv << i*dx << " " << j*dy << " " << k*dz << " " << V2(i,j,k) << " 0 0\n"; fw << i*dx << " " << j*dy << " " << k*dz << " " << V3(i,j,k) << " 0 0\n"; } }
    }
}

void help_command(){
	if(rank_mpi==0){
		cout<<"Usage: mpirun -np [N] ./src/fastSF.out [-s scalar] [-d 2D] ...\n";
	}
}

void get_Inputs(int argc, char* argv[]) {
    YAML::Node para; string para_path="in/para.yaml";
    try { para = YAML::LoadFile(para_path); }
    catch(YAML::Exception& e) { cerr << "Unable to open or parse '" + para_path + "': " << e.what() << endl; h5::finalize(); MPI_Finalize(); exit(1); }
    if (para["program"]["scalar_switch"]) scalar_switch = para["program"]["scalar_switch"].as<bool>();
    if (para["program"]["Only_longitudinal"]) longitudinal = para["program"]["Only_longitudinal"].as<bool>();
    if (para["program"]["2D_switch"]) two_dimension_switch = para["program"]["2D_switch"].as<bool>();
    if (para["program"]["Processors_X"]) px = para["program"]["Processors_X"].as<int>();
    if (para["test"]["test_switch"]) test_switch = para["test"]["test_switch"].as<bool>();
    if (test_switch){
    	if (para["grid"]["Nx"]) Nx = para["grid"]["Nx"].as<int>();
    	if (para["grid"]["Ny"]) Ny = para["grid"]["Ny"].as<int>();
    	if (para["grid"]["Nz"]) Nz = para["grid"]["Nz"].as<int>();
	}
    if (para["domain_dimension"]["Lx"]) Lx = para["domain_dimension"]["Lx"].as<double>();
    if (para["domain_dimension"]["Ly"]) Ly = para["domain_dimension"]["Ly"].as<double>();
    if (para["domain_dimension"]["Lz"]) Lz = para["domain_dimension"]["Lz"].as<double>();
    if (para["structure_function"]["q1"]) q1 = para["structure_function"]["q1"].as<int>();
    if (para["structure_function"]["q2"]) q2 = para["structure_function"]["q2"].as<int>();
    int option;
    while ((option=getopt(argc, argv, "X:Y:Z:1:2:x:y:z:l:d:p:t:s:U:V:W:Q:P:L:M:h:u:v:w:q:D"))!=-1){
    	switch(option){
    		case 'h': help_command(); exit(1); break;
            case 'D': dump_switch = true; break;
    		case 'X': Nx=std::stoi(optarg); break;
    		case 'Y': Ny=std::stoi(optarg); break;
    		case 'Z': Nz=std::stoi(optarg); break;
    		case 'x': Lx=std::stod(optarg); break;
    		case 'y': Ly=std::stod(optarg); break;
    		case 'z': Lz=std::stod(optarg); break;
    		case 'p': px=std::stoi(optarg); break;
    		case '1': q1=std::stoi(optarg); break;
    		case '2': q2=std::stoi(optarg); break;
    		case 't': test_switch=str_to_bool(optarg); break;
    		case 's': scalar_switch=str_to_bool(optarg); break;
    		case 'd': two_dimension_switch=str_to_bool(optarg); break;
    		case 'l': longitudinal=str_to_bool(optarg); break;
            case 'U': UName = optarg; break; case 'V': VName = optarg; break; case 'W': WName = optarg; break; case 'Q': TName = optarg; break;
            case 'u': UdName = optarg; break; case 'v': VdName = optarg; break; case 'w': WdName = optarg; break; case 'q': TdName = optarg; break;
            case 'P': SF_Grid_perp_name = optarg; break; case 'L': SF_Grid_pll_name = optarg; break; case 'M': SF_Grid_scalar_name = optarg; break;
    	}
    }
    // Smart defaults for TPP format: infer other names if only one is provided
    if (!scalar_switch) {
        if (UName != "U.V1r" && VName == "U.V2r") VName = UName;
        if (UName != "U.V1r" && WName == "U.V3r") WName = UName;
        if (UdName == "U.V1r" && UName != "U.V1r") UdName = "fields/vx";
        if (VdName == "U.V2r" && VName != "U.V2r") VdName = "fields/vy";
        if (WdName == "U.V3r" && WName != "U.V3r") WdName = "fields/vz";
    } else {
        if (TdName == "T.Fr" && TName != "T.Fr") TdName = "fields/temp";
    }
}
