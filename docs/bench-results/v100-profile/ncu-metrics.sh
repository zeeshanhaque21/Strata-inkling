#!/bin/bash
# arbitrary metrics for a kernel family: $1 regex, $2 launches, $3 csv, $4 comma list of metrics
export LD_LIBRARY_PATH=$HOME/cuda-12.9/lib CUDA_DEVICE_ORDER=PCI_BUS_ID INK2_Q8=15
. $HOME/driver580/v100-env.sh; cd $HOME/strata-inkling/runs
$HOME/nsight-compute/ncu --target-processes all --kernel-name "regex:$1" --launch-skip ${SKIP:-100} -c $2 --metrics $4 --csv --page raw --log-file $3 \
  $HOME/strata-bench/bench-bin/${BINNAME:-strata-ink2-volta5} --model /mnt/models/inkling-official --tokens-file $HOME/strata-inkling/runs/p24.ids --n 3 --ctx 4096 --kv k8v4 --chunk 512 --tc 1 --graph 0 > /dev/null 2>&1
echo "exit $?"
