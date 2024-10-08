import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser(description="Plot the timing results of the benchmarks")
parser.add_argument("--seq-length", action='store_true', help="Plot with seq length on X-axis", default=False)
args = parser.parse_args()

data = pd.read_csv(os.path.join("benchmarks", "timing_results.txt"))

sample_sizes = data["Sample Size"]
seq_lengths = data["Sequence Length"]
cpu_times = data["CPU Time (ms)"]
cpu_hashtable_times = data["CPU Hashtable Time (ms)"]
gpu_times = data["GPU Time (ms)"]
opt_gpu_times = data["Optimized GPU Time (ms)"]
gpu_hashtable_times = data["GPU Hashtable Time (ms)"]

x_ticks = seq_lengths if args.seq_length else sample_sizes
x_label = "Sequence Length" if args.seq_length else "Number of Samples"
title = "Performance of Different Implementations by Sequence Length" if args.seq_length \
   else "Performance of Different Implementations by Sample Size"
   

plt.figure(figsize=(10, 6))
plt.plot(x_ticks, cpu_times, label="CPU", marker='o')
plt.plot(x_ticks, cpu_hashtable_times, label="CPU with Hashtable", marker='o')
plt.plot(x_ticks, gpu_times, label="GPU", marker='o')
plt.plot(x_ticks, opt_gpu_times, label="Optimized GPU", marker='o')
plt.plot(x_ticks, gpu_hashtable_times, label="GPU with Hashtable", marker='o')

plt.xlabel(x_label)
plt.ylabel("Time (ms)")
plt.title(title)
plt.legend()
plt.grid(True)
plt.yscale("log")

plt.savefig(os.path.join("plots", f"performance_plot_{sample_sizes[0]}_{seq_lengths[0]}.png"))
