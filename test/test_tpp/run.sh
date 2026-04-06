#!/bin/bash
MPI_LAUNCHER="${MPI_LAUNCHER:-mpirun}"
MPI_NP_FLAG="${MPI_NP_FLAG:--np}"

if [ "$(uname)" = "Darwin" ]; then
    export FASTSF_SKIP_MPI_FINALIZE="${FASTSF_SKIP_MPI_FINALIZE:-1}"
fi

cd test/test_tpp
"$MPI_LAUNCHER" "$MPI_NP_FLAG" 1 ../../src/fastSF.out -U SampledData0 -u fields/vx -V SampledData0 -v fields/vy -W SampledData0 -w fields/vz
