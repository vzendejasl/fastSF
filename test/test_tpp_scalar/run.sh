#!/bin/bash
MPI_LAUNCHER="${MPI_LAUNCHER:-mpirun}"
MPI_NP_FLAG="${MPI_NP_FLAG:--np}"

if [ "$(uname)" = "Darwin" ]; then
    export FASTSF_SKIP_MPI_FINALIZE="${FASTSF_SKIP_MPI_FINALIZE:-1}"
fi

cd test/test_tpp_scalar
"$MPI_LAUNCHER" "$MPI_NP_FLAG" 1 ../../src/fastSF.out -Q ScalarTest -q fields/temp
