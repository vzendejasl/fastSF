#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ "$(uname)" = "Darwin" ]; then
    export FASTSF_SKIP_MPI_FINALIZE="${FASTSF_SKIP_MPI_FINALIZE:-1}"
fi

mkdir -p test_tpp_small/in test_tpp_small/out

# Ensure test/TPP_Small.h5 and TPP_Small_Scalar.h5 exist
python3 create_small_tpp.py test_tpp_small/in

cat > test_tpp_small/in/para.yaml <<EOF
program:
    scalar_switch: false
    2D_switch : false
    Only_longitudinal: false
    Processors_X: 1
domain_dimension :
    Lx : 1.0
    Ly : 1.0
    Lz : 1.0
structure_function :
    q1 : 1
    q2 : 2
test :
    test_switch : false
EOF

echo "Running Small TPP Velocity 3D Test..."
cd test_tpp_small
mpirun -np 1 ../../src/fastSF.out -U TPP_Small -u fields/vx -V TPP_Small -v fields/vy -W TPP_Small -w fields/vz > test_velocity_log.txt 2>&1
if grep -q "TOTAL KINETIC ENERGY (or Scalar SumSq): 3584" test_velocity_log.txt; then
    echo "Small TPP Velocity 3D Test PASSED"
else
    echo "Small TPP Velocity 3D Test FAILED"
    cat test_velocity_log.txt
    exit 1
fi

echo "Running Small TPP Scalar 3D Test..."
# Create a separate para.yaml for scalar test
cat > in/para.yaml <<EOF
program:
    scalar_switch: true
    2D_switch : false
    Only_longitudinal: false
    Processors_X: 1
domain_dimension :
    Lx : 1.0
    Ly : 1.0
    Lz : 1.0
structure_function :
    q1 : 1
    q2 : 2
test :
    test_switch : false
EOF

mpirun -np 1 ../../src/fastSF.out -Q TPP_Small_Scalar -q fields/temp > test_scalar_log.txt 2>&1
if grep -q "TOTAL KINETIC ENERGY (or Scalar SumSq): 512" test_scalar_log.txt; then
    echo "Small TPP Scalar 3D Test PASSED"
else
    echo "Small TPP Scalar 3D Test FAILED"
    cat test_scalar_log.txt
    exit 1
fi
cd ..

echo "All Small TPP Tests PASSED"
