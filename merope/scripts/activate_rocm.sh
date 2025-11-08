# bridge_rocm.sh
ATLAS_ROOT="$(realpath ../atlas)"
ROCM_BIN="$ATLAS_ROOT/build/dist/rocm/bin"
ROCM_LIB="$ATLAS_ROOT/build/dist/rocm/lib"
ROCM_PYTHON="$ATLAS_ROOT/build/dist/rocm/lib/python3.12/site-packages"

export PATH="$ROCM_BIN:$PATH"
export LD_LIBRARY_PATH="$ROCM_LIB:$LD_LIBRARY_PATH"
export PYTHONPATH="$ROCM_PYTHON:$PYTHONPATH"

echo "[Merope] Connected to ROCm SDK from Atlas"
