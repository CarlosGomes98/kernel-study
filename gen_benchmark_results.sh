#!/usr/bin/env bash

set -euo pipefail

# This scripts runs the ./benchmark binary for all existing kernels, and logs
# the outputs to text files in benchmark_results/. Then it calls
# the plotting script

mkdir -p benchmark_results

for kernel in {0..3}; do
    echo ""
    ./build/benchmark $kernel | tee "benchmark_results/${kernel}_output.txt"
    sleep 2
done

python3 plot_benchmark_results.py

