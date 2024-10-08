import matplotlib.pyplot as plt
import pandas as pd
import os

data = pd.read_csv(os.path.join("benchmarks", "timing_results.txt"))

sample_sizes = data["Sample Size"]
seq_length = data["Sequence Length"]
cpu_times = data["CPU Time (ms)"]
cpu_hashtable_times = data["CPU Hashtable Time (ms)"]
gpu_times = data["GPU Time (ms)"]
opt_gpu_times = data["Optimized GPU Time (ms)"]
gpu_hashtable_times = data["GPU Hashtable Time (ms)"]

plt.figure(figsize=(10, 6))
plt.plot(sample_sizes, cpu_times, label="CPU", marker='o')
plt.plot(sample_sizes, cpu_hashtable_times, label="CPU with Hashtable", marker='o')
plt.plot(sample_sizes, gpu_times, label="GPU", marker='o')
plt.plot(sample_sizes, opt_gpu_times, label="Optimized GPU", marker='o')
plt.plot(sample_sizes, gpu_hashtable_times, label="GPU with Hashtable", marker='o')

plt.xlabel("Number of Samples")
plt.ylabel("Time (ms)")
plt.title("Performance of Different Implementations by Sample Size")
plt.legend()
plt.grid(True)
plt.yscale("log")

plt.savefig(os.path.join("plots", f"performance_plot_{sample_sizes[0]}_{seq_length[0]}.png"))
