import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def plot_scaling(filename, title):
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return

    # Read the CSV (Columns: processors, samples, time, result)
    # We add header names because our CSV doesn't have them
    df = pd.read_csv(filename, names=["Cores", "Samples", "Time", "Result"])
    
    # Sort by number of cores just in case
    df = df.sort_values("Cores")

    # Calculate Speedup: Time(1 core) / Time(N cores)
    baseline_time = df[df["Cores"] == 1]["Time"].values[0]
    df["Speedup"] = baseline_time / df["Time"]
    
    # Calculate Efficiency: Speedup / Cores
    df["Efficiency"] = df["Speedup"] / df["Cores"]

    # --- Plot 1: Speedup ---
    plt.figure(figsize=(10, 5))
    plt.plot(df["Cores"], df["Speedup"], marker='o', label='Actual Speedup')
    plt.plot(df["Cores"], df["Cores"], 'r--', label='Ideal Speedup') # Perfect scaling line
    plt.title(f"Strong Scaling: {title}")
    plt.xlabel("Number of Processors")
    plt.ylabel("Speedup (Baseline = 1.0)")
    plt.grid(True)
    plt.legend()
    plt.savefig(filename.replace(".csv", "_speedup.png"))
    print(f"Saved {filename.replace('.csv', '_speedup.png')}")

    # --- Plot 2: Efficiency ---
    plt.figure(figsize=(10, 5))
    plt.plot(df["Cores"], df["Efficiency"], marker='s', color='green')
    plt.title(f"Parallel Efficiency: {title}")
    plt.xlabel("Number of Processors")
    plt.ylabel("Efficiency (1.0 = Perfect)")
    plt.ylim(0, 1.1)
    plt.grid(True)
    plt.savefig(filename.replace(".csv", "_efficiency.png"))
    print(f"Saved {filename.replace('.csv', '_efficiency.png')}")

if __name__ == "__main__":
    plot_scaling("results/scaling_data.csv", "Monte Carlo Pi")
    plot_scaling("results/options_data.csv", "Black-Scholes Options")