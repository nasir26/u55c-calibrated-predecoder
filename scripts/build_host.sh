#!/bin/bash
# Builds the XRT host program used by bench/run_card.py (runs/bin/xrt_predecoder) from host/xrt_predecoder.cpp.
# Needs XRT (source /opt/xilinx/xrt/setup.sh first). Author: Nasir Ali, C-DAC Noida.
set -euo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$R"
: "${XILINX_XRT:=/opt/xilinx/xrt}"
mkdir -p runs/bin
g++ -std=c++17 -O2 -Wall -I"$XILINX_XRT/include" host/xrt_predecoder.cpp -o "${1:-runs/bin/xrt_predecoder}" \
    -L"$XILINX_XRT/lib" -lxrt_coreutil -pthread
echo "built ${1:-runs/bin/xrt_predecoder}"
