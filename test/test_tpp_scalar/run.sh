#!/bin/bash
if [ "$(uname)" = "Darwin" ]; then
    export FASTSF_SKIP_MPI_FINALIZE="${FASTSF_SKIP_MPI_FINALIZE:-1}"
fi

cd test/test_tpp_scalar
mpirun -np 1 ../../src/fastSF.out -Q ScalarTest -q fields/temp
