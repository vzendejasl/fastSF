#!/bin/bash

#############################################################################################################################################
 # fastSF
 # 
 # Copyright (C) 2020, Mahendra K. Verma
 #
 # All rights reserved.
 # 
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions are met:
 #     1. Redistributions of source code must retain the above copyright
 #        notice, this list of conditions and the following disclaimer.
 #     2. Redistributions in binary form must reproduce the above copyright
 #        notice, this list of conditions and the following disclaimer in the
 #        documentation and/or other materials provided with the distribution.
 #     3. Neither the name of the copyright holder nor the
 #        names of its contributors may be used to endorse or promote products
 #        derived from this software without specific prior written permission.
 # 
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 # ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 # WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 # DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 # ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 # (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 # LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 # ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 # SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 #
 ############################################################################################################################################
 ##
 ##! \file runTest.sh
 #
 #   \brief Script to run the test case for validating fastSF
 #
 #   \author Shashwat Bhattacharya
 #   \date Feb 2020
 #   \copyright New BSD License
 #
 ############################################################################################################################################
##

if [ "$(uname)" = "Darwin" ]; then
    export FASTSF_SKIP_MPI_FINALIZE="${FASTSF_SKIP_MPI_FINALIZE:-1}"
    export MPLBACKEND="${MPLBACKEND:-Agg}"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
MPI_LAUNCHER="${MPI_LAUNCHER:-mpirun}"
MPI_NP_FLAG="${MPI_NP_FLAG:--np}"
MPI_NPROCS="${MPI_NPROCS:-1}"
FASTSF_EXECUTABLE="${FASTSF_EXECUTABLE:-../../src/fastSF.out}"

run_case() {
    test_dir="$1"
    cd "$test_dir"
    "$MPI_LAUNCHER" "$MPI_NP_FLAG" "$MPI_NPROCS" "$FASTSF_EXECUTABLE"
    cd ..
}

cd test
run_case test_scalar_2D
run_case test_velocity_2D

run_case test_scalar_3D
run_case test_velocity_3D
run_case test_velocity_3D_longitudinal

echo "Running TPP format tests..."
./test_tpp_all.sh

echo "Running sampled TXT/HDF5 input tests..."
bash test_input_conversion.sh

"$PYTHON_BIN" test.py
