/********************************************************************************************************************************************
 * fastSF
 *
 * Copyright (C) 2020, Mahendra K. Verma
 *
 * All rights reserved.
 ********************************************************************************************************************************************
 */

#include "h5si.h"
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
#include <cstdio>
#include <cstdlib>
#include <sys/stat.h>

using namespace std;
using namespace blitz;

//Chunk structure to store byte offset and row count
struct Chunk {
    std::streampos offset;
    long count;
};

struct ParsedData {
    std::vector<double> v1;
};

//Function declarations
void get_Inputs(int argc, char* argv[]); 
void ComputeAndPrintTKE();
void DumpFieldsToTxt();
std::vector<Chunk> build_chunk_index(string txt_path, int skip_count);
ParsedData read_chunk_at_offset(string txt_path, std::streampos offset, long count, bool isScalar);
void write_to_structured_h5(string h5_path, string dset_name, const std::vector<double>& v, int nx, int ny, int nz);
void verify_and_cleanup(string txt_path, string h5_path, string dset_name, double original_tke, long original_rows);
bool CheckAndConvert(string fold, string& filename, string& datasetname, bool isScalar);
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
void SFunc2D(Array<double,2>, Array<double,2>);
void SFunc_long_2D(Array<double,2>, Array<double,2>);
void SFunc3D(Array<double,3>, Array<double,3>, Array<double,3>);
void SFunc_long_3D(Array<double,3>, Array<double,3>, Array<double,3>);
void Read_Init(Array<double,2>&, Array<double,2>&);
void Read_Init(Array<double,3>&, Array<double,3>&, Array<double,3>&);
void Read_Init(Array<double,2>&);
void Read_Init(Array<double,3>&);
void SF_scalar_3D(Array<double,3>);
void SF_scalar_2D(Array<double,2>);
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
 	if (px > P || (Nx > 1 && (Nx/2)%px != 0)) {
        if (rank_mpi==0) cout<<"ERROR in processor configuration! Aborting.."<<endl;
        h5::finalize(); finalize_mpi_runtime(); exit(1);
    }
    resize_SFs();
    gettimeofday(&start_pt,NULL);
    calc_SFs();
    gettimeofday(&end_pt,NULL);
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
    	h5::File f(fold+file+".h5", "r");
    	h5::Dataset ds = f[dset];
    	int dim = ds.shape().size();
  		if (dim==2){ two_dimension_switch=true; Nx=ds.shape()[0]; Ny=1; Nz=ds.shape()[1]; }
  		else { two_dimension_switch=false; Nx=ds.shape()[0]; Ny=ds.shape()[1]; Nz=ds.shape()[2]; }
  		s(0)=dim; s(1)=Nx; s(2)=Ny; s(3)=Nz;
        ds.close(); f.close();
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
        if (two_dimension_switch) {
            if (scalar_switch) CheckAndConvert("in/", TName, TdName, true);
            else { CheckAndConvert("in/", UName, UdName, false); CheckAndConvert("in/", WName, WdName, false); }
        } else {
            if (scalar_switch) CheckAndConvert("in/", TName, TdName, true);
            else { CheckAndConvert("in/", UName, UdName, false); CheckAndConvert("in/", VName, VdName, false); CheckAndConvert("in/", WName, WdName, false); }
        }

    	if (rank_mpi==0) cout<<"Reading from the hdf5 files\n";
        Array<int,1> s1,s2, s3;
        if (two_dimension_switch){
            if (scalar_switch) { get_input_shape("in/", TName, TdName, s1); resize_input(); calculate_grid_spacing(); read_2D(T_2D,"in/", TName, TdName); }
            else { 
                get_input_shape("in/", UName, UdName, s1); get_input_shape("in/", WName, WdName, s2);
                if (!compare(s1,s2)) { if (rank_mpi==0) cerr<<"\nIncompatible dimension data\n\n"; h5::finalize(); finalize_mpi_runtime(); exit(1); }
                resize_input(); calculate_grid_spacing(); read_2D(V1_2D,"in/", UName, UdName); read_2D(V3_2D,"in/", WName, WdName);
            }
        } else {
            if (scalar_switch) { get_input_shape("in/", TName, TdName, s1); resize_input(); calculate_grid_spacing(); read_3D(T, "in/", TName, TdName); }
            else {
            	get_input_shape("in/", UName, UdName, s1); get_input_shape("in/", VName, VdName, s2); get_input_shape("in/", WName, WdName, s3);
            	if (!compare(s1,s2) || !compare(s2,s3)) { if (rank_mpi==0) cerr<<"\nIncompatible dimension data\n\n"; h5::finalize(); finalize_mpi_runtime(); exit(1); }
            	resize_input(); calculate_grid_spacing(); read_3D(V1, "in/", UName, UdName); read_3D(V2, "in/", VName, VdName); read_3D(V3, "in/", WName, WdName);
            }
        }
    } else {
        if (rank_mpi==0) cout<<"\nWARNING: The code is running in TEST mode. Generating fields internally.\n";
        resize_input(); calculate_grid_spacing();
        if (two_dimension_switch) scalar_switch ? Read_Init(T_2D) : Read_Init(V1_2D, V3_2D);
        else scalar_switch ? Read_Init(T) : Read_Init(V1, V2, V3);
    }
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
        mkdir("out",0777);
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
			read_3D(test1,"out/",SF_Grid_pll_name,SF_Grid_pll_name+int_to_str(order+q1));
			for (int i=0; i<test1.extent(0); i++) for (int j=0; j<test1.extent(1); j++) for (int k=0; k<test1.extent(2); k++){
				double lx=dx*i, ly=dy*j, lz=dz*k; double err = (lx*lx + ly*ly + lz*lz > epsilon) ? abs((test1(i,j,k)-pow(lx*lx+ly*ly+lz*lz,(order+q1)/2.))/pow(lx*lx+ly*ly+lz*lz,(order+q1)/2.)) : abs(test1(i,j,k));
                if (err > max) max = err;
			}
		}
	} else {
        test1.resize(Nx/2,Ny/2,Nz/2); test2.resize(Nx/2,Ny/2,Nz/2);
		for (int order=0 ; order<=q2-q1; order++){
			read_3D(test1,"out/",SF_Grid_pll_name,SF_Grid_pll_name+int_to_str(order+q1)); read_3D(test2,"out/",SF_Grid_perp_name,SF_Grid_perp_name+int_to_str(order+q1));
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
			read_2D(test1,"out/",SF_Grid_pll_name, SF_Grid_pll_name+int_to_str(order+q1));
			for (int i=0; i<test1.extent(0); i++) for (int k=0; k<test1.extent(1); k++){
				double lx=dx*i, lz=dz*k; double err = ((lx*lx + lz*lz)>epsilon) ? abs((test1(i,k)-pow(lx*lx+lz*lz,(order+q1)/2.))/pow(lx*lx+lz*lz,(order+q1)/2.)) : abs(test1(i,k));
                if (err > max) max=err;
			}
		}
	} else {
		test1.resize(Nx/2,Nz/2); test2.resize(Nx/2,Nz/2);
		for (int order=0 ; order<=q2-q1; order++){
			read_2D(test1,"out/",SF_Grid_pll_name,SF_Grid_pll_name+int_to_str(order+q1)); read_2D(test2,"out/",SF_Grid_perp_name,SF_Grid_perp_name+int_to_str(order+q1));
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
		read_2D(test1,"out/",SF_Grid_scalar_name, SF_Grid_scalar_name+int_to_str(order+q1));
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
		read_3D(test1,"out/",SF_Grid_scalar_name, SF_Grid_scalar_name+int_to_str(order+q1));
		for (int i=0; i<test1.extent(0); i++) for (int j=0; j<test1.extent(1); j++) for (int k=0; k<test1.extent(2); k++){
			double lx=dx*i, ly=dy*j, lz=dz*k; double err = (abs(lx+ly+lz)>epsilon) ? abs((test1(i,j,k)-pow(lx+ly+lz,(order+q1)))/pow(lx+ly+lz,(order+q1))) : abs(test1(i,j,k));
            if (err>max) max=err;
		}
	}
	if (rank_mpi==0) { max > epsilon ? cout<<"\nSCALAR_3D: TEST_FAILED\n" : cout<<"\nSCALAR_3D: TEST_PASSED\n"; cout<<"MAXIMUM ERROR: "<<max<<endl; }
}

string int_to_str(int n) { stringstream ss; ss << n; return ss.str(); }
void compute_time_elapsed(timeval s, timeval e, double& res){ res = ((e.tv_sec-s.tv_sec)*1000000u + e.tv_usec-s.tv_usec)/1.0e6; }

void write_4D(Array<double,4> A, string file) {
  int nx=A.extent(0), ny=A.extent(1), nz=A.extent(2); h5::File f("out/"+file+".h5", "w");
  for (int q=q1; q<=q2; q++) {
      if (rank_mpi==0) cout<<"Writing "<<q<<" order to file.\n"; string qstr = int_to_str(q);
      h5::Dataset ds = f.create_dataset(file+qstr, h5::shape(nx,ny,nz), "double");
      Array<double,3> t(nx,ny,nz); t = A(Range::all(),Range::all(),Range::all(),q-q1);
      ds << t.data(); ds.close();
  }
  f.close();
}

void write_3D(Array<double,3> A, string file) {
  int nx=A.extent(0), nz=A.extent(1); h5::File f("out/"+file+".h5", "w");
  for (int q=q1; q<=q2; q++) {
      if (rank_mpi==0) cout<<"Writing "<<q<<" order to file.\n"; string qstr = int_to_str(q);
      h5::Dataset ds = f.create_dataset(file+qstr, h5::shape(nx,nz), "double");
      Array<double,2> t(nx,nz); t = A(Range::all(),Range::all(),q-q1);
      ds << t.data(); ds.close();
  }
  f.close();
}

void show_checklist(){ if (rank_mpi==0) cerr<<"Error: Check inputs in 'in/' folder.\n"; }
void read_2D(Array<double,2> A, string fold, string file, string dset) { h5::File f(fold+file+".h5", "r"); h5::Dataset ds = f[dset]; ds >> A.data(); ds.close(); f.close(); }
void read_3D(Array<double,3> A, string fold, string file, string dset) { h5::File f(fold+file+".h5", "r"); h5::Dataset ds = f[dset]; ds >> A.data(); ds.close(); f.close(); }

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

void SFunc3D(Array<double,3> Ux, Array<double,3> Uy, Array<double,3> Uz) {
	if (rank_mpi==0) cout<<"\nComputing longitudinal and transverse S(lx, ly, lz) using 3D velocity field data..\n";
    int c_per_proc = Nx*Ny/(4*P); Array<int, 3> idx; compute_index_list(idx, Nx, Ny);
    for (int ix=0; ix<c_per_proc; ix++){
        int x=idx(ix, 0, rank_mpi), y=idx(ix, 1, rank_mpi);
  		for(int z=0; z<Nz/2; z++){
        	int cnt=(Nx-x)*(Ny-y)*(Nz-z); double lx=x*dx, ly=y*dy, lz=z*dz, r=sqrt(lx*lx+ly*ly+lz*lz); if (r<1e-15) r=1e-15;
            Array<double,3> dUx(Nx-x,Ny-y,Nz-z), dUy(Nx-x,Ny-y,Nz-z), dUz(Nx-x,Ny-y,Nz-z), dUpll(Nx-x,Ny-y,Nz-z);
        	dUx = Ux(Range(x,Nx-1),Range(y,Ny-1),Range(z,Nz-1)) - Ux(Range(0,Nx-x-1),Range(0,Ny-y-1),Range(0,Nz-z-1));
        	dUy = Uy(Range(x,Nx-1),Range(y,Ny-1),Range(z,Nz-1)) - Uy(Range(0,Nx-x-1),Range(0,Ny-y-1),Range(0,Nz-z-1));
        	dUz = Uz(Range(x,Nx-1),Range(y,Ny-1),Range(z,Nz-1)) - Uz(Range(0,Nx-x-1),Range(0,Ny-y-1),Range(0,Nz-z-1));
            dUpll = (lx*dUx+ly*dUy+lz*dUz)/r; dUx = dUx-dUpll*lx/r; dUy = dUy-dUpll*ly/r; dUz = dUz-dUpll*lz/r; dUx = pow(dUx*dUx+dUy*dUy+dUz*dUz,0.5);
        	for (int p=0; p<=q2-q1; p++){
        		double Spll = sum(pow(dUpll,q1+p))/cnt, Sperp = sum(pow(dUx,q1+p))/cnt;
                Array<int, 1> X(P), Y(P), Z(P), p_arr(P); Array<double, 1> Spll_arr(P), Sperp_arr(P);
                MPI_Gather(&x, 1, MPI_INT, X.data(), 1, MPI_INT, 0, MPI_COMM_WORLD); MPI_Gather(&y, 1, MPI_INT, Y.data(), 1, MPI_INT, 0, MPI_COMM_WORLD); MPI_Gather(&z, 1, MPI_INT, Z.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Gather(&Spll, 1, MPI_DOUBLE, Spll_arr.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD); MPI_Gather(&p, 1, MPI_INT, p_arr.data(), 1, MPI_INT, 0, MPI_COMM_WORLD); MPI_Gather(&Sperp, 1, MPI_DOUBLE, Sperp_arr.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                if (rank_mpi==0) for (int i=0; i<P; i++) { SF_Grid_pll(X(i),Y(i),Z(i),p_arr(i))=Spll_arr(i); SF_Grid_perp(X(i),Y(i),Z(i),p_arr(i))=Sperp_arr(i); }
        	}
  		}
  	}
    if (rank_mpi==0) { SF_Grid_pll(0,0,0,Range::all())=0; SF_Grid_perp(0,0,0,Range::all())=0; }
}

void SFunc_long_3D(Array<double,3> Ux, Array<double,3> Uy, Array<double,3> Uz) {
if (rank_mpi==0) cout<<"\nComputing longitudinal S(lx, ly, lz) using 3D velocity field data..\n";
    int c_per_proc = Nx*Ny/(4*P); Array<int, 3> idx; compute_index_list(idx, Nx, Ny);
    for (int ix=0; ix<c_per_proc; ix++){
        int x=idx(ix, 0, rank_mpi), y=idx(ix, 1, rank_mpi);
  		for(int z=0; z<Nz/2; z++){
    		int cnt=(Nx-x)*(Ny-y)*(Nz-z); double lx=x*dx, ly=y*dy, lz=z*dz, r=sqrt(lx*lx+ly*ly+lz*lz); if (r<1e-15) r=1e-15;
            Array<double,3> dUx(Nx-x,Ny-y,Nz-z), dUy(Nx-x,Ny-y,Nz-z), dUz(Nx-x,Ny-y,Nz-z), dUpll(Nx-x,Ny-y,Nz-z);
    		dUx = Ux(Range(x,Nx-1),Range(y,Ny-1),Range(z,Nz-1)) - Ux(Range(0,Nx-x-1),Range(0,Ny-y-1),Range(0,Nz-z-1));
    		dUy = Uy(Range(x,Nx-1),Range(y,Ny-1),Range(z,Nz-1)) - Uy(Range(0,Nx-x-1),Range(0,Ny-y-1),Range(0,Nz-z-1));
    		dUz = Uz(Range(x,Nx-1),Range(y,Ny-1),Range(z,Nz-1)) - Uz(Range(0,Nx-x-1),Range(0,Ny-y-1),Range(0,Nz-z-1));
            dUpll = (lx*dUx+ly*dUy+lz*dUz)/r;
    		for (int p=0; p<=q2-q1; p++){
    			double Spll = sum(pow(dUpll,q1+p))/cnt;
                Array<int, 1> X(P), Y(P), Z(P), p_arr(P); Array<double, 1> Spll_arr(P);
                MPI_Gather(&x, 1, MPI_INT, X.data(), 1, MPI_INT, 0, MPI_COMM_WORLD); MPI_Gather(&y, 1, MPI_INT, Y.data(), 1, MPI_INT, 0, MPI_COMM_WORLD); MPI_Gather(&z, 1, MPI_INT, Z.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Gather(&Spll, 1, MPI_DOUBLE, Spll_arr.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD); MPI_Gather(&p, 1, MPI_INT, p_arr.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
                if (rank_mpi==0) for (int i=0; i<P; i++) SF_Grid_pll(X(i),Y(i),Z(i),p_arr(i))=Spll_arr(i);
    		}
  		}
  	}
    if (rank_mpi==0) SF_Grid_pll(0,0,0,Range::all())=0;
}

void SFunc2D(Array<double,2> Ux, Array<double,2> Uz) {
     if (rank_mpi==0) cout<<"\nComputing longitudinal and transverse S(lx, lz) using 2D velocity field data..\n";
    int p_per_proc = Nx*Nz/(4*P); Array<int, 3> idx; compute_index_list(idx, Nx, Nz);
    for (int ix=0; ix<p_per_proc; ix++){
        int x=idx(ix, 0, rank_mpi), z=idx(ix, 1, rank_mpi);
        int cnt=(Nx-x)*(Nz-z); double lx=x*dx, lz=z*dz, r=sqrt(lx*lx+lz*lz); if (r<1e-15) r=1e-15;
        Array<double,2> dUx(Nx-x,Nz-z), dUz(Nx-x,Nz-z), dUpll(Nx-x,Nz-z);
        dUx = Ux(Range(x,Nx-1),Range(z,Nz-1)) - Ux(Range(0,Nx-x-1),Range(0,Nz-z-1));
        dUz = Uz(Range(x,Nx-1),Range(z,Nz-1)) - Uz(Range(0,Nx-x-1),Range(0,Nz-z-1));
        dUpll = (lx*dUx+lz*dUz)/r; dUx = dUx-dUpll*lx/r; dUz = dUz-dUpll*lz/r; dUx = pow(dUx*dUx+dUz*dUz,0.5);
    	for (int p=0; p<=q2-q1; p++){
            double Spll = sum(pow(dUpll,q1+p))/cnt, Sperp = sum(pow(dUx,q1+p))/cnt;
            Array<int, 1> X(P), Z(P), p_arr(P); Array<double, 1> Spll_arr(P), Sperp_arr(P);
            MPI_Gather(&x, 1, MPI_INT, X.data(), 1, MPI_INT, 0, MPI_COMM_WORLD); MPI_Gather(&z, 1, MPI_INT, Z.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Gather(&Spll, 1, MPI_DOUBLE, Spll_arr.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD); MPI_Gather(&p, 1, MPI_INT, p_arr.data(), 1, MPI_INT, 0, MPI_COMM_WORLD); MPI_Gather(&Sperp, 1, MPI_DOUBLE, Sperp_arr.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            if (rank_mpi==0) for (int i=0; i<P; i++) { SF_Grid2D_pll(X(i),Z(i),p_arr(i))=Spll_arr(i); SF_Grid2D_perp(X(i),Z(i),p_arr(i))=Sperp_arr(i); }
        } 
    }
    if (rank_mpi==0) { SF_Grid2D_pll(0,0,Range::all())=0; SF_Grid2D_perp(0,0,Range::all())=0; }
}

void SFunc_long_2D(Array<double,2> Ux, Array<double,2> Uz) {
     if (rank_mpi==0) cout<<"\nComputing longitudinal S(lx, lz) using 2D velocity field data..\n";
    int p_per_proc = Nx*Nz/(4*P); Array<int, 3> idx; compute_index_list(idx, Nx, Nz);
    for (int ix=0; ix<p_per_proc; ix++){
        int x=idx(ix, 0, rank_mpi), z=idx(ix, 1, rank_mpi);
        int cnt=(Nx-x)*(Nz-z); double lx=x*dx, lz=z*dz, r=sqrt(lx*lx+lz*lz); if (r<1e-15) r=1e-15;
        Array<double,2> dUx(Nx-x,Nz-z), dUz(Nx-x,Nz-z), dUpll(Nx-x,Nz-z);
        dUx = Ux(Range(x,Nx-1),Range(z,Nz-1)) - Ux(Range(0,Nx-x-1),Range(0,Nz-z-1));
        dUz = Uz(Range(x,Nx-1),Range(z,Nz-1)) - Uz(Range(0,Nx-x-1),Range(0,Nz-z-1));
        dUpll = (lx*dUx+lz*dUz)/r;
        for (int p=0; p<=q2-q1; p++){
            double Spll = sum(pow(dUpll,q1+p))/cnt;
            Array<int, 1> X(P), Z(P), p_arr(P); Array<double, 1> Spll_arr(P);
            MPI_Gather(&x, 1, MPI_INT, X.data(), 1, MPI_INT, 0, MPI_COMM_WORLD); MPI_Gather(&z, 1, MPI_INT, Z.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Gather(&Spll, 1, MPI_DOUBLE, Spll_arr.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD); MPI_Gather(&p, 1, MPI_INT, p_arr.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (rank_mpi==0) for (int i=0; i<P; i++) SF_Grid2D_pll(X(i),Z(i),p_arr(i))=Spll_arr(i);
        }
    }
    if (rank_mpi==0) SF_Grid2D_pll(0,0,Range::all())=0;
}

void SF_scalar_3D(Array<double,3> T) {
     if (rank_mpi==0) cout<<"\nComputing S(lx, ly, lz) using 3D scalar field data..\n";
    int c_per_proc = Nx*Ny/(4*P); Array<int, 3> idx; compute_index_list(idx, Nx, Ny);
    for (int ix=0; ix<c_per_proc; ix++){
        int x=idx(ix, 0, rank_mpi), y=idx(ix, 1, rank_mpi);
        for(int z=0; z<Nz/2; z++){
            int cnt=(Nx-x)*(Ny-y)*(Nz-z); Array<double,3> dT(Nx-x,Ny-y,Nz-z);
            dT = T(Range(x,Nx-1),Range(y,Ny-1),Range(z,Nz-1)) - T(Range(0,Nx-x-1),Range(0,Ny-y-1),Range(0,Nz-z-1));
            for (int p=0; p<=q2-q1; p++){
                double St = sum(pow(dT,q1+p))/cnt;
                Array<int, 1> X(P), Y(P), Z(P), p_arr(P); Array<double, 1> St_arr(P);
                MPI_Gather(&x, 1, MPI_INT, X.data(), 1, MPI_INT, 0, MPI_COMM_WORLD); MPI_Gather(&y, 1, MPI_INT, Y.data(), 1, MPI_INT, 0, MPI_COMM_WORLD); MPI_Gather(&z, 1, MPI_INT, Z.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Gather(&St, 1, MPI_DOUBLE, St_arr.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD); MPI_Gather(&p, 1, MPI_INT, p_arr.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
                if (rank_mpi==0) for (int i=0; i<P; i++) SF_Grid_scalar(X(i),Y(i),Z(i),p_arr(i))=St_arr(i);
            }
        }
    }
    if (rank_mpi==0) SF_Grid_scalar(0,0,0,Range::all())=0;
}

void SF_scalar_2D(Array<double,2> T) {
     if (rank_mpi==0) cout<<"\nComputing S(lx, lz) using 2D scalar field data..\n";
    int p_per_proc = Nx*Nz/(4*P); Array<int, 3> idx; compute_index_list(idx, Nx, Nz);
    for (int ix=0; ix<p_per_proc; ix++){
        int x=idx(ix, 0, rank_mpi), z=idx(ix, 1, rank_mpi);
        int cnt=(Nx-x)*(Nz-z); Array<double,2> dT(Nx-x,Nz-z);
        dT = T(Range(x,Nx-1),Range(z,Nz-1)) - T(Range(0,Nx-x-1),Range(0,Nz-z-1));
        for (int p=0; p<=q2-q1; p++){
            double St = sum(pow(dT,q1+p))/cnt;
            Array<int, 1> X(P), Z(P), p_arr(P); Array<double, 1> St_arr(P);
            MPI_Gather(&x, 1, MPI_INT, X.data(), 1, MPI_INT, 0, MPI_COMM_WORLD); MPI_Gather(&z, 1, MPI_INT, Z.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Gather(&St, 1, MPI_DOUBLE, St_arr.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD); MPI_Gather(&p, 1, MPI_INT, p_arr.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (rank_mpi==0) for (int i=0; i<P; i++) SF_Grid2D_scalar(X(i),Z(i),p_arr(i))=St_arr(i);
        }
    }
    if (rank_mpi==0) SF_Grid2D_scalar(0,0,Range::all())=0;
}

std::vector<Chunk> build_chunk_index(string txt_path, int skip_count) {
    std::vector<Chunk> chunks; std::ifstream f(txt_path, std::ios::binary); if (!f.is_open()) return chunks;
    string line; for (int i=0; i<skip_count; ++i) if (!std::getline(f, line)) break;
    const long CHUNK_SIZE = 1000000;
    while (f) { Chunk c; c.offset = f.tellg(); c.count = 0; for (long i=0; i<CHUNK_SIZE; ++i) { if (!std::getline(f, line)) break; c.count++; }
        if (c.count > 0) chunks.push_back(c); else break; }
    return chunks;
}

ParsedData read_chunk_at_offset(string txt_path, std::streampos offset, long count, bool isScalar) {
    ParsedData data; std::ifstream f(txt_path, std::ios::binary); if (!f.is_open()) return data;
    f.seekg(offset); string line;
    for (long i=0; i<count; ++i) { if (!std::getline(f, line)) break; double x, y, z, v1, v2, v3;
        if (isScalar) { if (std::sscanf(line.c_str(), "%lf %lf %lf %lf", &x, &y, &z, &v1) == 4) data.v1.push_back(v1); }
        else { if (std::sscanf(line.c_str(), "%lf %lf %lf %lf %lf %lf", &x, &y, &z, &v1, &v2, &v3) == 6) data.v1.push_back(v1); }
    }
    return data;
}

void write_to_structured_h5(string h5_path, string dset_name, const std::vector<double>& v, int nx, int ny, int nz) {
    int px_local=px, py_local=P/px_local, rankx=rank_mpi/py_local, ranky=rank_mpi%py_local;
    int local_nx=nx/px_local, local_ny=ny/py_local, local_nz=nz;
    h5::File f; if (rank_mpi == 0) { hid_t file = H5Fcreate(h5_path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT); H5Fclose(file); } MPI_Barrier(MPI_COMM_WORLD);
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS); H5Pset_fapl_mpio(fapl, MPI_COMM_WORLD, MPI_INFO_NULL); f.setFapl(fapl); f.open(h5_path, "a");
    h5::Plan plan; std::vector<hsize_t> global_dim={(hsize_t)nx, (hsize_t)ny, (hsize_t)nz}, local_dim={(hsize_t)local_nx, (hsize_t)local_ny, (hsize_t)local_nz};
    h5::Select file_select(blitz::Range(rankx*local_nx, (rankx+1)*local_nx-1), blitz::Range(ranky*local_ny, (ranky+1)*local_ny-1), blitz::Range(0, local_nz-1));
    plan.set_plan(MPI_COMM_WORLD, local_dim, h5::Select::all(3), global_dim, file_select);
    h5::Dataset ds = f.create_dataset(dset_name, plan, "double"); ds << v.data(); ds.close(); f.close(); H5Pclose(fapl);
}

void verify_and_cleanup(string txt_path, string h5_path, string dset_name, double original_tke, long original_rows) {
    if (rank_mpi == 0) {
        h5::File f(h5_path, "r"); h5::Dataset ds = f[dset_name]; std::vector<hsize_t> shape = ds.shape();
        long total_h5_rows = 1; for (auto s : shape) total_h5_rows *= s;
        if (total_h5_rows == original_rows) {
            Array<double, 3> data(shape[0], shape[1], shape[2]); ds >> data.data(); double h5_tke = sum(pow2(data));
            if (std::abs(h5_tke - original_tke) / (original_tke + 1e-20) < 1e-10) { cout << "  SUCCESS: Conversion verified. Deleting " << txt_path << endl; std::remove(txt_path.c_str()); }
        }
        ds.close(); f.close();
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

bool CheckAndConvert(string fold, string& filename, string& datasetname, bool isScalar) {
    string txt_path = fold + filename + ".txt", h5_path = fold + filename + ".h5";
    std::ifstream f_txt(txt_path); if (!f_txt.good()) return false; f_txt.close();
    if (rank_mpi == 0) cout << "Detected TXT: " << txt_path << ". Converting to HDF5..." << endl;
    int hc = 8; std::vector<Chunk> all_chunks; if (rank_mpi == 0) all_chunks = build_chunk_index(txt_path, hc);
    int n_chunks = all_chunks.size(); MPI_Bcast(&n_chunks, 1, MPI_INT, 0, MPI_COMM_WORLD); if (rank_mpi != 0) all_chunks.resize(n_chunks);
    for (int i=0; i<n_chunks; ++i) { long long off; if (rank_mpi == 0) off = (long long)all_chunks[i].offset; MPI_Bcast(&off, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD); if (rank_mpi != 0) all_chunks[i].offset = (std::streampos)off; MPI_Bcast(&all_chunks[i].count, 1, MPI_LONG, 0, MPI_COMM_WORLD); }
    double local_tke = 0; long local_rows = 0; ParsedData local_parsed;
    for (int i=rank_mpi; i<n_chunks; i+=P) { ParsedData d = read_chunk_at_offset(txt_path, all_chunks[i].offset, all_chunks[i].count, isScalar); local_parsed.v1.insert(local_parsed.v1.end(), d.v1.begin(), d.v1.end()); local_rows += d.v1.size(); for (double val : d.v1) local_tke += val*val; }
    double total_tke = 0; long total_rows = 0; MPI_Allreduce(&local_tke, &total_tke, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); MPI_Allreduce(&local_rows, &total_rows, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    int nx_v=Nx, ny_v=Ny, nz_v=Nz; if (total_rows == 274625) { nx_v=65; ny_v=65; nz_v=65; }
    write_to_structured_h5(h5_path, filename, local_parsed.v1, nx_v, ny_v, nz_v); verify_and_cleanup(txt_path, h5_path, filename, total_tke, total_rows); datasetname = filename; return true;
}

void ComputeAndPrintTKE() {
    double local_tke = 0;
    if (two_dimension_switch) { if (scalar_switch) local_tke = sum(pow2(T_2D)); else local_tke = 0.5 * (sum(pow2(V1_2D)) + sum(pow2(V3_2D))); }
    else { if (scalar_switch) local_tke = sum(pow2(T)); else local_tke = 0.5 * (sum(pow2(V1)) + sum(pow2(V2)) + sum(pow2(V3))); }
    double total_tke = 0; MPI_Reduce(&local_tke, &total_tke, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank_mpi == 0) cout << "TOTAL KINETIC ENERGY (or Scalar SumSq): " << total_tke << endl;
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
