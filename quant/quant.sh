#!/usr/bin/env bash

set -euo pipefail

OUTDIR="quant"
LOGFILE="${OUTDIR}/quant.txt"

# Make sure output directory exists
mkdir -p "$OUTDIR"

# Clear previous log
: > "$LOGFILE"

run() {
  echo "========================================" | tee -a "$LOGFILE"
  echo "Running: $*" | tee -a "$LOGFILE"
  echo "Started at: $(date)" | tee -a "$LOGFILE"
  echo "----------------------------------------" | tee -a "$LOGFILE"

  "$@" >> "$LOGFILE" 2>&1

  echo "Finished at: $(date)" | tee -a "$LOGFILE"
  echo >> "$LOGFILE"
}
run nsys profile -o quant/quant1  python scripts/profile_kernel_bandwidth.py --warmup 100 --iters 100 --batch-size 1  --tokens 1024 --hidden-dim 3072 --fp4
run nsys profile -o quant/quant2  python scripts/profile_kernel_bandwidth.py --warmup 100 --iters 100 --batch-size 1  --tokens 2048 --hidden-dim 3072 --fp4

run nsys profile -o quant/quant3  python scripts/profile_kernel_bandwidth.py --warmup 100 --iters 100 --batch-size 1  --tokens 4096 --hidden-dim 3072 --fp4
run nsys profile -o quant/quant4  python scripts/profile_kernel_bandwidth.py --warmup 100 --iters 100 --batch-size 1  --tokens 8192 --hidden-dim 3072 --fp4
run nsys profile -o quant/quant5  python scripts/profile_kernel_bandwidth.py --warmup 100 --iters 100 --batch-size 1  --tokens 4096 --hidden-dim 2048 --fp4
run nsys profile -o quant/quant6  python scripts/profile_kernel_bandwidth.py --warmup 100 --iters 100 --batch-size 1  --tokens 4096 --hidden-dim 4096 --fp4
run nsys profile -o quant/quant7  python scripts/profile_kernel_bandwidth.py --warmup 100 --iters 100 --batch-size 1  --tokens 4096 --hidden-dim 8192 --fp4

run nsys profile -o quant/quant8  python scripts/profile_kernel_bandwidth.py --warmup 100 --iters 100 --batch-size 4  --tokens 2048 --hidden-dim 3072 --fp4
run nsys profile -o quant/quant9  python scripts/profile_kernel_bandwidth.py --warmup 100 --iters 100 --batch-size 4  --tokens 4096 --hidden-dim 3072 --fp4 
run nsys profile -o quant/quant10  python scripts/profile_kernel_bandwidth.py --warmup 100 --iters 100 --batch-size 4  --tokens 8192 --hidden-dim 3072 --fp4

run nsys profile -o quant/quant11  python scripts/profile_kernel_bandwidth.py --warmup 100 --iters 100 --batch-size 16 --tokens 4096 --hidden-dim 3072 --fp4
run nsys profile -o quant/quant12 python scripts/profile_kernel_bandwidth.py --warmup 100 --iters 100 --batch-size 16 --tokens 8192 --hidden-dim 3072 --fp4
run nsys profile -o quant/quant13 python scripts/profile_kernel_bandwidth.py --warmup 100 --iters 100 --batch-size 64 --tokens 4096 --hidden-dim 3072 --fp4
run nsys profile -o quant/quant14 python scripts/profile_kernel_bandwidth.py --warmup 100 --iters 100 --batch-size 64 --tokens 8192 --hidden-dim 3072 --fp4
run nsys profile -o quant/quant15 python scripts/profile_kernel_bandwidth.py --warmup 100 --iters 100 --batch-size 16 --tokens 4096 --hidden-dim 8192 --fp4

echo "All runs completed successfully." | tee -a "$LOGFILE"
