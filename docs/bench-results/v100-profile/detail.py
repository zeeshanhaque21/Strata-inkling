#!/usr/bin/env python3
"""Print the Nsight Compute 'details' CSV of a few launches of one kernel: speed of light, scheduler, warp-state and memory metrics.  usage: detail.py file.csv [launch ids...]"""
import csv, io, sys
lines = [l for l in open(sys.argv[1], errors="ignore") if not l.startswith("==")]
rows = list(csv.reader(io.StringIO("".join(lines)))); h = rows[0]; ix = {c: i for i, c in enumerate(h)}
ids = sys.argv[2:] or ["0", "1"]
keep = ("Duration", "DRAM Throughput", "Memory Throughput", "Compute (SM) Throughput", "L1/TEX Cache Throughput", "L2 Cache Throughput", "Executed Ipc Active", "Issue Slots Busy",
        "Warp Cycles Per Issued Instruction", "Warp Cycles Per Executed Instruction", "Active Warps Per Scheduler", "Eligible Warps Per Scheduler", "No Eligible", "Issued Warp Per Scheduler",
        "Achieved Occupancy", "Theoretical Occupancy", "Registers Per Thread", "Executed Instructions", "SM Busy", "Max Bandwidth", "Mem Busy", "Mem Pipes Busy", "L1/TEX Hit Rate",
        "Block Limit Registers", "Block Limit Shared Mem", "Block Limit Warps", "Waves Per SM")
for kid in ids:
    ks = [r for r in rows[1:] if len(r) > 14 and r[0] == kid]
    if not ks: continue
    print(f"\n=== launch {kid}: {ks[0][ix['Kernel Name']][:60]}  grid {ks[0][ix['Grid Size']]} block {ks[0][ix['Block Size']]}")
    for r in ks:
        if r[ix["Metric Name"]] in keep: print(f"  {r[ix['Section Name']][:24]:24s} {r[ix['Metric Name']]:38s} {r[ix['Metric Value']]:>14s} {r[ix['Metric Unit']]}")
