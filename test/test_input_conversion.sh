#!/bin/bash
set -euo pipefail

if [ "$(uname)" = "Darwin" ]; then
    export FASTSF_SKIP_MPI_FINALIZE="${FASTSF_SKIP_MPI_FINALIZE:-1}"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CASE_DIR="$REPO_ROOT/test_input_case"

rm -rf "$CASE_DIR"
trap 'rm -rf "$CASE_DIR"' EXIT
mkdir -p "$CASE_DIR/in" "$CASE_DIR/out"
python3 "$SCRIPT_DIR/create_sampled_txt.py" "$CASE_DIR/in"

cat > "$CASE_DIR/in/para.yaml" <<'EOF'
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

cd "$CASE_DIR"

echo "Running sampled velocity TXT conversion test..."
mpirun -np 2 "$REPO_ROOT/src/fastSF.out" -U SampledDataTxt > velocity_txt_log.txt 2>&1
grep -q "Detected TXT" velocity_txt_log.txt
grep -q "TOTAL KINETIC ENERGY (or Scalar SumSq): 3584" velocity_txt_log.txt
[ ! -f in/SampledDataTxt.txt ]
[ -f in/SampledDataTxt.h5 ]
python3 - <<'PY'
import h5py
hf = h5py.File("in/SampledDataTxt.h5", "r")
schema = hf.attrs["schema"]
if isinstance(schema, bytes):
    schema = schema.decode("utf-8")
assert schema == "structured_velocity_v1"
assert tuple(hf["fields"]["vx"].shape) == (8, 8, 8)
assert tuple(hf["fields"]["vy"].shape) == (8, 8, 8)
assert tuple(hf["fields"]["vz"].shape) == (8, 8, 8)
assert tuple(hf["grid"]["x"].shape) == (8,)
assert tuple(hf["grid"]["y"].shape) == (8,)
assert tuple(hf["grid"]["z"].shape) == (8,)
hf.close()
PY

echo "Running shuffled sampled velocity TXT conversion test..."
mpirun -np 4 "$REPO_ROOT/src/fastSF.out" -U SampledDataTxt_shuffled > velocity_txt_shuffled_log.txt 2>&1
grep -q "TOTAL KINETIC ENERGY (or Scalar SumSq): 3584" velocity_txt_shuffled_log.txt
[ ! -f in/SampledDataTxt_shuffled.txt ]
[ -f in/SampledDataTxt_shuffled.h5 ]

echo "Running periodic sampled velocity TXT conversion test..."
mpirun -np 2 "$REPO_ROOT/src/fastSF.out" -U PeriodicSampledDataTxt > periodic_velocity_txt_log.txt 2>&1
grep -q "TOTAL KINETIC ENERGY (or Scalar SumSq): 3584" periodic_velocity_txt_log.txt
[ ! -f in/PeriodicSampledDataTxt.txt ]
[ -f in/PeriodicSampledDataTxt.h5 ]
python3 - <<'PY'
import h5py
hf = h5py.File("in/PeriodicSampledDataTxt.h5", "r")
assert tuple(hf["fields"]["vx"].shape) == (8, 8, 8)
assert tuple(hf["grid"]["x"].shape) == (8,)
hf.close()
PY

echo "Running structured velocity HDF5 fast-path test..."
mpirun -np 2 "$REPO_ROOT/src/fastSF.out" -U StructuredVelocity > velocity_h5_log.txt 2>&1
grep -q "TOTAL KINETIC ENERGY (or Scalar SumSq): 3584" velocity_h5_log.txt
if grep -q "Detected TXT" velocity_h5_log.txt; then
    echo "Structured HDF5 path unexpectedly triggered TXT conversion"
    exit 1
fi

echo "Running periodic structured velocity HDF5 fast-path test..."
mpirun -np 2 "$REPO_ROOT/src/fastSF.out" -U PeriodicStructuredVelocity > periodic_velocity_h5_log.txt 2>&1
grep -q "TOTAL KINETIC ENERGY (or Scalar SumSq): 3584" periodic_velocity_h5_log.txt
if grep -q "Detected TXT" periodic_velocity_h5_log.txt; then
    echo "Periodic structured HDF5 path unexpectedly triggered TXT conversion"
    exit 1
fi

cat > in/para.yaml <<'EOF'
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

echo "Running sampled scalar TXT conversion test..."
mpirun -np 2 "$REPO_ROOT/src/fastSF.out" -Q SampledScalar > scalar_txt_log.txt 2>&1
grep -q "Detected TXT" scalar_txt_log.txt
grep -q "TOTAL KINETIC ENERGY (or Scalar SumSq): 512" scalar_txt_log.txt
[ ! -f in/SampledScalar.txt ]
[ -f in/SampledScalar.h5 ]
python3 - <<'PY'
import h5py
hf = h5py.File("in/SampledScalar.h5", "r")
schema = hf.attrs["schema"]
if isinstance(schema, bytes):
    schema = schema.decode("utf-8")
assert schema == "structured_scalar_v1"
assert tuple(hf["fields"]["temp"].shape) == (8, 8, 8)
hf.close()
PY

echo "Running periodic sampled scalar TXT conversion test..."
mpirun -np 2 "$REPO_ROOT/src/fastSF.out" -Q PeriodicSampledScalar > periodic_scalar_txt_log.txt 2>&1
grep -q "TOTAL KINETIC ENERGY (or Scalar SumSq): 512" periodic_scalar_txt_log.txt
[ ! -f in/PeriodicSampledScalar.txt ]
[ -f in/PeriodicSampledScalar.h5 ]
python3 - <<'PY'
import h5py
hf = h5py.File("in/PeriodicSampledScalar.h5", "r")
assert tuple(hf["fields"]["temp"].shape) == (8, 8, 8)
hf.close()
PY

echo "Running structured scalar HDF5 fast-path test..."
mpirun -np 2 "$REPO_ROOT/src/fastSF.out" -Q StructuredScalar > scalar_h5_log.txt 2>&1
grep -q "TOTAL KINETIC ENERGY (or Scalar SumSq): 512" scalar_h5_log.txt
if grep -q "Detected TXT" scalar_h5_log.txt; then
    echo "Structured scalar HDF5 path unexpectedly triggered TXT conversion"
    exit 1
fi

echo "Running periodic structured scalar HDF5 fast-path test..."
mpirun -np 2 "$REPO_ROOT/src/fastSF.out" -Q PeriodicStructuredScalar > periodic_scalar_h5_log.txt 2>&1
grep -q "TOTAL KINETIC ENERGY (or Scalar SumSq): 512" periodic_scalar_h5_log.txt
if grep -q "Detected TXT" periodic_scalar_h5_log.txt; then
    echo "Periodic structured scalar HDF5 path unexpectedly triggered TXT conversion"
    exit 1
fi

echo "All input conversion tests PASSED"
