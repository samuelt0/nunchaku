#!/usr/bin/env bash

set -euo pipefail

OUTDIR="mul"
LOGFILE="${OUTDIR}/mul.txt"

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
run nsys profile -o mul/mul1 python scripts/profile_mul_add_kernel_bandwidth.py --warmup 100 --iters 100 --batch-size 1  --tokens 1024 --hidden-dim 3072
run nsys profile -o mul/mul2 python scripts/profile_mul_add_kernel_bandwidth.py --warmup 100 --iters 100 --batch-size 1  --tokens 2048 --hidden-dim 3072

run nsys profile -o mul/mul3 python scripts/profile_mul_add_kernel_bandwidth.py --warmup 100 --iters 100 --batch-size 1  --tokens 4096 --hidden-dim 3072
run nsys profile -o mul/mul4  python scripts/profile_mul_add_kernel_bandwidth.py --warmup 100 --iters 100 --batch-size 1  --tokens 8192 --hidden-dim 3072
run nsys profile -o mul/mul5  python scripts/profile_mul_add_kernel_bandwidth.py --warmup 100 --iters 100 --batch-size 1  --tokens 4096 --hidden-dim 2048
run nsys profile -o mul/mul6  python scripts/profile_mul_add_kernel_bandwidth.py --warmup 100 --iters 100 --batch-size 1  --tokens 4096 --hidden-dim 4096
run nsys profile -o mul/mul7  python scripts/profile_mul_add_kernel_bandwidth.py --warmup 100 --iters 100 --batch-size 1  --tokens 4096 --hidden-dim 8192

run nsys profile -o mul/mul8  python scripts/profile_mul_add_kernel_bandwidth.py --warmup 100 --iters 100 --batch-size 4  --tokens 2048 --hidden-dim 3072
run nsys profile -o mul/mul9 python scripts/profile_mul_add_kernel_bandwidth.py --warmup 100 --iters 100 --batch-size 4  --tokens 4096 --hidden-dim 3072
run nsys profile -o mul/mul10 python scripts/profile_mul_add_kernel_bandwidth.py --warmup 100 --iters 100 --batch-size 4  --tokens 8192 --hidden-dim 3072

run nsys profile -o mul/mul11  python scripts/profile_mul_add_kernel_bandwidth.py --warmup 100 --iters 100 --batch-size 16 --tokens 4096 --hidden-dim 3072
run nsys profile -o mul/mul12 python scripts/profile_mul_add_kernel_bandwidth.py --warmup 100 --iters 100 --batch-size 16 --tokens 8192 --hidden-dim 3072
run nsys profile -o mul/mul13 python scripts/profile_mul_add_kernel_bandwidth.py --warmup 100 --iters 100 --batch-size 64 --tokens 4096 --hidden-dim 3072
run nsys profile -o mul/mul14 python scripts/profile_mul_add_kernel_bandwidth.py --warmup 100 --iters 100 --batch-size 64 --tokens 8192 --hidden-dim 3072
run nsys profile -o mul/mul15 python scripts/profile_mul_add_kernel_bandwidth.py --warmup 100 --iters 100 --batch-size 16 --tokens 4096 --hidden-dim 8192

echo "All runs completed successfully." | tee -a "$LOGFILE"
