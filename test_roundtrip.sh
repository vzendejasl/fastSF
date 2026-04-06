#!/bin/bash
set -euo pipefail

export DYLD_LIBRARY_PATH=$HOME/local/lib:${DYLD_LIBRARY_PATH:-}

echo "==============================================================="
echo "ROUNDTRIP TEST: Sampled TXT -> structured HDF5 -> fastSF"
echo "==============================================================="

bash test/test_input_conversion.sh

echo "==============================================================="
echo "ROUNDTRIP TESTS PASSED"
echo "==============================================================="
