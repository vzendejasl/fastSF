# Building fastSF on macOS

This guide details the local environment setup and build process used on this Mac.

## 1. Prerequisites (Homebrew)

The core dependencies were installed via Homebrew:

```bash
brew install yaml-cpp blitz open-mpi hdf5
```

## 2. Local Dependencies (H5SI)

The `h5si` library was built manually and installed into a local folder in the home directory (`$HOME/local`).

```bash
# Clone and enter the repository
git clone https://github.com/anandogc/h5si.git
cd h5si/trunk

# Create build directory
mkdir build && cd build

# Configure with MPI enabled
# Note: Use the local prefix for installation
cmake .. \
  -DCMAKE_CXX_COMPILER=mpicxx \
  -DCMAKE_C_COMPILER=mpicc \
  -DCMAKE_INSTALL_PREFIX=$HOME/local \
  -DH5SI_ENABLE_MPI=ON

# Build and install
make install
```

## 3. Environment Variables

To ensure the compiler and linker can find the local `h5si` and Homebrew libraries, the following variables are used:

```bash
export PREFIX=$HOME/local
export PATH=$PREFIX/bin:$PATH
export CPATH=$PREFIX/include:$CPATH
export LIBRARY_PATH=$PREFIX/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$PREFIX/lib:$LD_LIBRARY_PATH

# pkg-config helps the Makefile find Homebrew's yaml-cpp and blitz
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
```

## 4. Compiling fastSF

The `fastSF` Makefile is designed to use the `PREFIX` variable to locate the local `h5si` installation and `pkg-config` for the rest.

```bash
cd src
make PREFIX=$HOME/local
```

## 5. Running with MPI on Mac

On this machine, a specific environment variable is often required to avoid issues with `MPI_Finalize` during cleanup:

```bash
export FASTSF_SKIP_MPI_FINALIZE=1
mpirun -np 1 ./fastSF.out -U SampledData0
```
