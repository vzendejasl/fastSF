# Building fastSF on Tuolumne

This guide captures the Tuolumne build path that was validated for this repository. It keeps all locally built dependencies in one place and uses the GNU-flavored Cray HDF5 install that matches the `gcc/13.3.1-magic` and `cray-mpich/9.1.0` toolchain.

## 1. Validated Environment

Load the Tuolumne modules first. `StdEnv` must be loaded before the `*-magic` compiler stack.

```bash
module --force purge
module load StdEnv
module load gcc/13.3.1-magic
module load cray-mpich/9.1.0
module load cray-hdf5-parallel/1.14.3.7
module load cmake/3.29.2
```

Use one sibling directory for local dependencies:

```bash
export FASTSF_HOME=/g/g11/zendejas/Documents/fastSF
export DEP_HOME=/g/g11/zendejas/Documents/fastSF_deps
export PREFIX=$DEP_HOME/install
export HDF5_ROOT=/opt/cray/pe/hdf5-parallel/1.14.3.7/gnu/12.2

mkdir -p "$DEP_HOME/repos" "$DEP_HOME/build" "$PREFIX" "$DEP_HOME/bin"

export PATH="$DEP_HOME/bin:$PREFIX/bin:$PATH"
export CPATH="$PREFIX/include:$CPATH"
export LIBRARY_PATH="$PREFIX/lib:$PREFIX/lib64:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/opt/cray/pe/lib64/cce:$HDF5_ROOT/lib:$PREFIX/lib:$PREFIX/lib64:$LD_LIBRARY_PATH"
export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig:$PREFIX/lib64/pkgconfig:$PREFIX/share/pkgconfig:$PKG_CONFIG_PATH"
```

Blitz expects a `python` executable during `make install`. On Tuolumne, `python3` is usually available but `python` may not be, so create a local shim once:

```bash
ln -sf "$(command -v python3)" "$DEP_HOME/bin/python"
```

## 2. Build Local Dependencies

### Blitz++ 1.0.2

```bash
cd "$DEP_HOME/repos"
if [ ! -d blitz ]; then
  git clone https://github.com/blitzpp/blitz.git
fi

cd blitz
git fetch --tags
git checkout tags/1.0.2
./configure --prefix="$PREFIX" CC=gcc CXX=g++
make -j8 install
```

Verify:

```bash
pkg-config --modversion blitz
pkg-config --libs blitz
```

### yaml-cpp

```bash
cd "$DEP_HOME/repos"
if [ ! -d yaml-cpp ]; then
  git clone https://github.com/jbeder/yaml-cpp.git
fi

cd yaml-cpp
git fetch --tags
rm -rf build

cmake -S . -B build \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DYAML_CPP_BUILD_TESTS=OFF \
  -DCMAKE_C_COMPILER=gcc \
  -DCMAKE_CXX_COMPILER=g++

cmake --build build -j8
cmake --install build
```

Verify:

```bash
pkg-config --modversion yaml-cpp
pkg-config --variable=pcfiledir yaml-cpp
```

### H5SI

`h5si` needs two small source fixes on Tuolumne:

- modern CMake is stricter about the minimum version declaration
- the upstream code calls the old HDF5 `H5Oget_info_by_name` signature, but Tuolumne's HDF5 exports the versioned API

Patch and build it against the GNU HDF5 variant under the Cray install:

```bash
cd "$DEP_HOME/repos"
if [ ! -d h5si ]; then
  git clone https://github.com/anandogc/h5si.git
fi

cd h5si/trunk
sed -i 's/CMAKE_MINIMUM_REQUIRED(VERSION 2.8.2)/CMAKE_MINIMUM_REQUIRED(VERSION 3.5)/' CMakeLists.txt

python3 - <<'EOF'
from pathlib import Path

patches = [
    (
        Path("lib/h5node.cc"),
        "H5Oget_info_by_name(parent->id(), name.c_str(), &this->info_, H5P_DEFAULT);",
        "H5Oget_info_by_name(parent->id(), name.c_str(), &this->info_, H5O_INFO_BASIC, H5P_DEFAULT);",
    ),
    (
        Path("lib/h5dataset.cc"),
        "        this->plan = dataset.plan;\n    }",
        "        this->plan = dataset.plan;\n        return *this;\n    }",
    ),
]

for path, old, new in patches:
    text = path.read_text()
    if old in text:
        path.write_text(text.replace(old, new, 1))
EOF

rm -rf build

cmake -S . -B build \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_C_COMPILER=mpicc \
  -DCMAKE_CXX_COMPILER=mpicxx \
  -DFIND_LIBRARIES=OFF \
  -DENABLE_PARALLEL=TRUE \
  -DCACHED_INCLUDES="$PREFIX/include;$HDF5_ROOT/include" \
  -DCACHED_LIBRARY_DIRS="$PREFIX/lib;$HDF5_ROOT/lib" \
  -DCACHED_LIBRARIES="blitz;hdf5" \
  -DCMAKE_CXX_FLAGS="-DH5_HAVE_PARALLEL" \
  -Wno-dev

cmake --build build --target h5si -j8
cmake --install build
```

Verify:

```bash
nm -C "$PREFIX/lib/libh5si.a" | egrep 'h5::shape\(|H5Oget_info_by_name' | head -20
```

The expected symbols are `h5::shape(unsigned long, ...)` and `H5Oget_info_by_name3`.

## 3. Build fastSF

```bash
cd "$FASTSF_HOME/src"
make PREFIX="$PREFIX" \
     H5SI_ROOT="$PREFIX" \
     HDF5_ROOT="$HDF5_ROOT" \
     MPICXX=mpicxx
```

This creates `fastSF.out` in `src/`.

## 4. Run Tests

### Quickest first test

The fastest Tuolumne sanity check is the small TPP test:

```bash
cd "$FASTSF_HOME/test"
MPI_LAUNCHER=srun MPI_NP_FLAG=-n bash test_tpp_all.sh
```

To override the rank count for that script:

```bash
cd "$FASTSF_HOME/test"
MPI_LAUNCHER=srun MPI_NP_FLAG=-n TPP_TEST_NPROCS=2 bash test_tpp_all.sh
```

This runs one-rank velocity and scalar checks and should print:

```text
Small TPP Velocity 3D Test PASSED
Small TPP Scalar 3D Test PASSED
All Small TPP Tests PASSED
```

This test needs `python3`, `numpy`, and `h5py`.

### Input conversion tests

After the small TPP test passes, run the sampled TXT and HDF5 input tests:

```bash
cd "$FASTSF_HOME/test"
MPI_LAUNCHER=srun MPI_NP_FLAG=-n bash test_input_conversion.sh
```

To override ranks for the conversion cases, set one or more of these variables:

```bash
INPUT_VELOCITY_TXT_NPROCS=2
INPUT_VELOCITY_SHUFFLED_NPROCS=4
INPUT_PERIODIC_VELOCITY_TXT_NPROCS=2
INPUT_VELOCITY_H5_NPROCS=2
INPUT_PERIODIC_VELOCITY_H5_NPROCS=2
INPUT_SCALAR_TXT_NPROCS=2
INPUT_PERIODIC_SCALAR_TXT_NPROCS=2
INPUT_SCALAR_H5_NPROCS=2
INPUT_PERIODIC_SCALAR_H5_NPROCS=2
```

Example:

```bash
cd "$FASTSF_HOME/test"
MPI_LAUNCHER=srun MPI_NP_FLAG=-n \
INPUT_VELOCITY_TXT_NPROCS=2 \
INPUT_VELOCITY_SHUFFLED_NPROCS=4 \
INPUT_SCALAR_TXT_NPROCS=2 \
bash test_input_conversion.sh
```

This also needs `python3`, `numpy`, and `h5py`.

### Full test suite

Run the repository test driver from the repo root:

```bash
cd "$FASTSF_HOME"
MPI_LAUNCHER=srun MPI_NP_FLAG=-n PYTHON_BIN=python3 bash runTest.sh
```

`runTest.sh` executes:

- the four analytic regression tests
- `test/test_tpp_all.sh`
- `test/test_input_conversion.sh`
- `test/test.py`, which generates plots

For the full suite, you also need `matplotlib` in addition to `numpy` and `h5py`.

## 5. Run fastSF

For a TPP-style structured velocity file such as `SampledData0.h5`:

```bash
cd "$FASTSF_HOME"
srun -n 4 ./src/fastSF.out -U SampledData0
```

This auto-detects the standard `fields/vx`, `fields/vy`, and `fields/vz` layout.

## 6. Copy-Paste Test Script

If you already have a Tuolumne node allocation, you can use this helper script to reload the validated environment and run the quickest regression test.

```bash
#!/bin/bash
set -euo pipefail

module --force purge
module load StdEnv
module load gcc/13.3.1-magic
module load cray-mpich/9.1.0
module load cray-hdf5-parallel/1.14.3.7
module load cmake/3.29.2

export FASTSF_HOME=/g/g11/zendejas/Documents/fastSF
export DEP_HOME=/g/g11/zendejas/Documents/fastSF_deps
export PREFIX=$DEP_HOME/install
export HDF5_ROOT=/opt/cray/pe/hdf5-parallel/1.14.3.7/gnu/12.2

mkdir -p "$DEP_HOME/bin"
export PATH="$DEP_HOME/bin:$PREFIX/bin:$PATH"
export CPATH="$PREFIX/include:$CPATH"
export LIBRARY_PATH="$PREFIX/lib:$PREFIX/lib64:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/opt/cray/pe/lib64/cce:$HDF5_ROOT/lib:$PREFIX/lib:$PREFIX/lib64:$LD_LIBRARY_PATH"
export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig:$PREFIX/lib64/pkgconfig:$PREFIX/share/pkgconfig:$PKG_CONFIG_PATH"

ln -sf "$(command -v python3)" "$DEP_HOME/bin/python"

cd "$FASTSF_HOME/test"
MPI_LAUNCHER=srun MPI_NP_FLAG=-n TPP_TEST_NPROCS=1 bash test_tpp_all.sh
```

If you save that as `test_fastsf_tuolumne.sh`, run it with:

```bash
bash test_fastsf_tuolumne.sh
```

If you already have a node checked out and your environment is still loaded, the shortest command to run right now is:

```bash
cd /g/g11/zendejas/Documents/fastSF/test
MPI_LAUNCHER=srun MPI_NP_FLAG=-n TPP_TEST_NPROCS=1 bash test_tpp_all.sh
```
