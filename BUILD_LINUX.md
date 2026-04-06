# Building fastSF on Linux

This guide provides step-by-step instructions for building `fastSF` and its required dependencies on a standard Linux system. All dependencies are installed into a local directory to keep your system environment clean.

## 1. Prerequisites
Ensure your system has the following basic tools installed:
*   `gcc`, `g++`, `make`, `cmake`, `git`, `wget`
*   An MPI implementation (e.g., `MPICH` or `OpenMPI`)
*   `autoconf`, `libtool` (required for Blitz++)

## 2. Set Up Environment
Run these commands to create a local dependency folder and set the necessary environment variables. It is recommended to add these to your `~/.bashrc` or a setup script.

```bash
# Define your installation directory (e.g., inside the project folder)
export BASE_DIR=$(pwd)
export PREFIX=$BASE_DIR/deps/install
mkdir -p $BASE_DIR/deps/build $PREFIX

# Update paths to use the local dependencies
export PATH=$PREFIX/bin:$PATH
export CPATH=$PREFIX/include:$CPATH
export LIBRARY_PATH=$PREFIX/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$PREFIX/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=$PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH
```

---

## 3. Install Dependencies

### A. Blitz++ (v1.0.2)
```bash
cd $BASE_DIR/deps/build
git clone https://github.com/blitzpp/blitz.git
cd blitz && git checkout tags/1.0.2
autoreconf -fiv
./configure --prefix=$PREFIX CXX=g++ CC=gcc
make -j$(nproc) install
```

### B. YAML-cpp
```bash
cd $BASE_DIR/deps/build
git clone https://github.com/jbeder/yaml-cpp.git
cd yaml-cpp && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$PREFIX -DYAML_CPP_BUILD_TESTS=OFF
make -j$(nproc) install
```

### C. Parallel HDF5 (v1.8.20)
*Note: Using version 1.8.20 is recommended for maximum compatibility.*
```bash
cd $BASE_DIR/deps/build
wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.8/hdf5-1.8.20/src/hdf5-1.8.20.tar.gz
tar -xvf hdf5-1.8.20.tar.gz
cd hdf5-1.8.20
CC=mpicc CXX=mpicxx ./configure --prefix=$PREFIX --enable-parallel --without-zlib
make -j$(nproc) install
```

### D. H5SI
H5SI requires a small patch to its `CMakeLists.txt` for modern CMake versions:
```bash
cd $BASE_DIR/deps/build
git clone https://github.com/anandogc/h5si.git
cd h5si/trunk
# Update CMake requirement to at least 3.5
sed -i 's/CMAKE_MINIMUM_REQUIRED(VERSION 2.8.2)/CMAKE_MINIMUM_REQUIRED(VERSION 3.5)/' CMakeLists.txt

mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$PREFIX \
         -DCMAKE_CXX_COMPILER=mpicxx \
         -DCMAKE_C_COMPILER=mpicc \
         -DHDF5_ROOT=$PREFIX \
         -DHDF5_IS_PARALLEL=TRUE \
         -DCMAKE_POLICY_DEFAULT_CMP0074=NEW
make -j$(nproc) install
```

---

## 4. Build fastSF
Now that the dependencies are ready, compile the main application:

```bash
cd $BASE_DIR/src
make MPICXX=mpicxx PREFIX=$PREFIX
```
This will generate the executable `fastSF.out` in the `src/` directory.

---

## 5. Verification
Run the automated test suite to ensure everything is working correctly:
```bash
cd $BASE_DIR
export LD_LIBRARY_PATH=$PREFIX/lib:$LD_LIBRARY_PATH
bash runTest.sh
```
