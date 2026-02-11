#!/usr/bin/env bash

set -euo pipefail

OUTDIR="layer"
LOGFILE="${OUTDIR}/layer.txt"

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
run nsys profile -o layer/layer1  python scripts/profile_layernorm_kernel_bandwidth.py --warmup 100 --iters 100 --batch-size 1  --tokens 1024 --hidden-dim 3072
run nsys profile -o layer/layer2  python scripts/profile_layernorm_kernel_bandwidth.py --warmup 100 --iters 100 --batch-size 1  --tokens 2048 --hidden-dim 3072

run nsys profile -o layer/layer3  python scripts/profile_layernorm_kernel_bandwidth.py --warmup 100 --iters 100 --batch-size 1  --tokens 4096 --hidden-dim 3072
run nsys profile -o layer/layer4  python scripts/profile_layernorm_kernel_bandwidth.py --warmup 100 --iters 100 --batch-size 1  --tokens 8192 --hidden-dim 3072
run nsys profile -o layer/layer5  python scripts/profile_layernorm_kernel_bandwidth.py --warmup 100 --iters 100 --batch-size 1  --tokens 4096 --hidden-dim 2048
run nsys profile -o layer/layer6  python scripts/profile_layernorm_kernel_bandwidth.py --warmup 100 --iters 100 --batch-size 1  --tokens 4096 --hidden-dim 4096
run nsys profile -o layer/layer7  python scripts/profile_layernorm_kernel_bandwidth.py --warmup 100 --iters 100 --batch-size 1  --tokens 4096 --hidden-dim 8192

run nsys profile -o layer/layer8  python scripts/profile_layernorm_kernel_bandwidth.py --warmup 100 --iters 100 --batch-size 4  --tokens 2048 --hidden-dim 3072
run nsys profile -o layer/layer9 python scripts/profile_layernorm_kernel_bandwidth.py --warmup 100 --iters 100 --batch-size 4  --tokens 4096 --hidden-dim 3072
run nsys profile -o layer/layer10 python scripts/profile_layernorm_kernel_bandwidth.py --warmup 100 --iters 100 --batch-size 4  --tokens 8192 --hidden-dim 3072

run nsys profile -o layer/layer11  python scripts/profile_layernorm_kernel_bandwidth.py --warmup 100 --iters 100 --batch-size 16 --tokens 4096 --hidden-dim 3072
run nsys profile -o layer/layer12 python scripts/profile_layernorm_kernel_bandwidth.py --warmup 100 --iters 100 --batch-size 16 --tokens 8192 --hidden-dim 3072
run nsys profile -o layer/layer13 python scripts/profile_layernorm_kernel_bandwidth.py --warmup 100 --iters 100 --batch-size 64 --tokens 4096 --hidden-dim 3072
run nsys profile -o layer/layer14 python scripts/profile_layernorm_kernel_bandwidth.py --warmup 100 --iters 100 --batch-size 64 --tokens 8192 --hidden-dim 3072
run nsys profile -o layer/layer15 python scripts/profile_layernorm_kernel_bandwidth.py --warmup 100 --iters 100 --batch-size 16 --tokens 4096 --hidden-dim 8192

echo "All runs completed successfully." | tee -a "$LOGFILE"
