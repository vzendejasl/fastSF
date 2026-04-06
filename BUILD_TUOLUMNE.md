# Building fastSF on Tuolumne

This guide details how to build `fastSF` and its dependencies on the Tuolumne cluster.

## 1. Environment Setup

Load the necessary system modules and set up your local installation paths.

```bash
# Load Tuolumne specific modules
module purge
module load gcc/13.3.1-magic
module load cray-mpich/9.1.0
module load cray-hdf5-parallel/1.14.3.7
module load cmake/3.29.2

# Define the single directory for all dependencies
export DEP_HOME=/g/g11/zendejas/Documents/fluidSF_dependencies
export PREFIX=$DEP_HOME/install

mkdir -p $DEP_HOME/repos
mkdir -p $PREFIX

# Export paths for compilers and runtime
export PATH=$PREFIX/bin:$PATH
export CPATH=$PREFIX/include:$CPATH
export LIBRARY_PATH=$PREFIX/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$PREFIX/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=$PREFIX/lib/pkgconfig:$PREFIX/share/pkgconfig:$PKG_CONFIG_PATH
```

---

## 2. Install Dependencies

### A. Blitz++ (v1.0.2)
```bash
cd $DEP_HOME/repos
git clone https://github.com/blitzpp/blitz.git
cd blitz
git checkout tags/1.0.2
./configure --prefix=$PREFIX
make -j 8 install
```

### B. YAML-cpp
```bash
cd $DEP_HOME/repos
git clone https://github.com/jbeder/yaml-cpp.git
cd yaml-cpp
mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$PREFIX -DYAML_CPP_BUILD_TESTS=OFF
make -j 8 install
```

### C. H5SI (with Parallel HDF5 support)
Note: We must explicitly force the `H5_HAVE_PARALLEL` flag for the compiler to see the MPI-enabled HDF5 functions.

```bash
cd $DEP_HOME/repos
git clone https://github.com/anandogc/h5si.git
cd h5si/trunk  # Enter the source directory
mkdir -p build && cd build
cmake .. \
  -DCMAKE_CXX_COMPILER=mpicxx \
  -DCMAKE_C_COMPILER=mpicc \
  -DCMAKE_INSTALL_PREFIX=$PREFIX \
  -DCMAKE_CXX_FLAGS="-DH5_HAVE_PARALLEL" \
  -DHDF5_PREFER_PARALLEL=ON
make -j 8 install
```

---

## 3. Compile fastSF

Navigate back to your project directory and run make.

```bash
cd /g/g11/zendejas/Documents/fastSF/src
make PREFIX=$PREFIX MPICXX=mpicxx
```

---

## 4. Usage & Running

### Automated Setup Script
You can create a script named `setup_fastSF.sh` to quickly load your environment:

```bash
#!/bin/bash
module load gcc/13.3.1-magic cray-mpich/9.1.0 cray-hdf5-parallel/1.14.3.7 cmake/3.29.2
export PREFIX=/g/g11/zendejas/Documents/fluidSF_dependencies/install
export LD_LIBRARY_PATH=$PREFIX/lib:$LD_LIBRARY_PATH
```

### Streamlined TPP Command
If your data is in the `turbulence_post_process` format (e.g., `SampledData0.h5`), you can run it with a single flag:

```bash
source setup_fastSF.sh
mpirun -np 4 ./src/fastSF.out -U SampledData0
```
This will automatically infer `fields/vx`, `fields/vy`, and `fields/vz` from the single file.
