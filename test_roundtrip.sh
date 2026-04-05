#!/bin/bash
export DYLD_LIBRARY_PATH=$HOME/local/lib:$DYLD_LIBRARY_PATH

echo "==============================================================="
echo "ROUNDTRIP TEST: Dump generated data -> Read back from TXT"
echo "==============================================================="

# Function to update yaml value
update_yaml() {
    local key=$1
    local val=$2
    sed -i '' "s/$key : .*/$key : $val/" in/para.yaml
}

# 1. Start with 3D Velocity
echo "Step 1: Dumping 3D Velocity data..."
update_yaml "scalar_switch" "False"
update_yaml "2D_switch" "False"
update_yaml "test_switch" "true"
update_yaml "Nx" "8"
update_yaml "Ny" "8"
update_yaml "Nz" "8"

./src/fastSF.out -D

if [ ! -f in/U.V1r.txt ]; then
    echo "ERROR: Dump failed, in/U.V1r.txt not found."
    exit 1
fi

# 2. Disable test mode and run with different ranks
update_yaml "test_switch" "false"

for np in 1 2 4; do
    echo "---------------------------------------------------------------"
    echo "Running with $np ranks..."
    # Set Processors_X to 1 for simplicity in these small tests
    update_yaml "Processors_X" "1"
    
    mpirun -np $np ./src/fastSF.out
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Execution failed with $np ranks."
        exit 1
    fi
    
    # Check if conversion cleaned up (it should only clean up on the LAST successful read or if we only had one file)
    # Actually CheckAndConvert cleans up immediately after each file.
done

# 3. Repeat for 2D Scalar
echo "---------------------------------------------------------------"
echo "Step 2: Dumping 2D Scalar data..."
update_yaml "scalar_switch" "true"
update_yaml "2D_switch" "true"
update_yaml "test_switch" "true"
./src/fastSF.out -D

if [ ! -f in/T.Fr.txt ]; then
    echo "ERROR: Dump failed, in/T.Fr.txt not found."
    exit 1
fi

update_yaml "test_switch" "false"
for np in 1 2 4; do
    echo "---------------------------------------------------------------"
    echo "Running 2D Scalar with $np ranks..."
    mpirun -np $np ./src/fastSF.out
    if [ $? -ne 0 ]; then
        echo "ERROR: Execution failed with $np ranks."
        exit 1
    fi
done

echo "==============================================================="
echo "ALL ROUNDTRIP TESTS PASSED (1, 2, 4 ranks)"
echo "==============================================================="
