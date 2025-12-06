import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import sys

def plot_strong_scaling(df, title, output_prefix):
    """Generate strong scaling plots (speedup and efficiency)."""
    # Sort by number of cores
    df = df.sort_values("ranks")

    # Calculate Speedup: Time(1 core) / Time(N cores)
    baseline_time = df[df["ranks"] == 1]["time_sec"].values
    if len(baseline_time) == 0:
        print(f"Warning: No 1-rank baseline found for {title}. Cannot calculate speedup.")
        return
    
    baseline_time = baseline_time[0]
    df["speedup"] = baseline_time / df["time_sec"]
    
    # Calculate Efficiency: Speedup / Cores
    df["efficiency"] = df["speedup"] / df["ranks"]

    # --- Plot 1: Speedup ---
    plt.figure(figsize=(10, 6))
    plt.plot(df["ranks"], df["speedup"], marker='o', linewidth=2, markersize=8, label='Actual Speedup')
    plt.plot(df["ranks"], df["ranks"], 'r--', linewidth=2, label='Ideal (Linear) Speedup')
    plt.title(f"Strong Scaling: {title}", fontsize=14, fontweight='bold')
    plt.xlabel("Number of MPI Ranks", fontsize=12)
    plt.ylabel("Speedup (Baseline = 1.0)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    speedup_file = f"{output_prefix}_speedup.png"
    plt.savefig(speedup_file, dpi=300)
    print(f"✓ Saved: {speedup_file}")
    plt.close()

    # --- Plot 2: Efficiency ---
    plt.figure(figsize=(10, 6))
    plt.plot(df["ranks"], df["efficiency"], marker='s', linewidth=2, markersize=8, color='green')
    plt.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Ideal Efficiency')
    plt.axhline(y=0.7, color='orange', linestyle=':', linewidth=1.5, label='70% Threshold')
    plt.title(f"Parallel Efficiency: {title}", fontsize=14, fontweight='bold')
    plt.xlabel("Number of MPI Ranks", fontsize=12)
    plt.ylabel("Efficiency (1.0 = Perfect)", fontsize=12)
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    efficiency_file = f"{output_prefix}_efficiency.png"
    plt.savefig(efficiency_file, dpi=300)
    print(f"✓ Saved: {efficiency_file}")
    plt.close()
    
    # Print summary statistics
    print(f"\n{title} - Strong Scaling Summary:")
    print(f"{'Ranks':<8} {'Time (s)':<12} {'Speedup':<10} {'Efficiency':<12}")
    print("-" * 45)
    for _, row in df.iterrows():
        print(f"{int(row['ranks']):<8} {row['time_sec']:<12.4f} {row['speedup']:<10.2f} {row['efficiency']:<12.2%}")
    print()

def plot_weak_scaling(df, title, output_prefix):
    """Generate weak scaling plots (time should stay constant)."""
    df = df.sort_values("ranks")
    
    # For weak scaling, we expect time to remain roughly constant
    baseline_time = df[df["ranks"] == 1]["time_sec"].values
    if len(baseline_time) == 0:
        print(f"Warning: No 1-rank baseline found for {title}. Cannot calculate weak scaling efficiency.")
        return
    
    baseline_time = baseline_time[0]
    df["scaled_efficiency"] = baseline_time / df["time_sec"]
    
    # --- Plot 1: Execution Time vs Ranks ---
    plt.figure(figsize=(10, 6))
    plt.plot(df["ranks"], df["time_sec"], marker='o', linewidth=2, markersize=8, color='blue')
    plt.axhline(y=baseline_time, color='r', linestyle='--', linewidth=2, label=f'Baseline ({baseline_time:.2f}s)')
    plt.title(f"Weak Scaling: {title}", fontsize=14, fontweight='bold')
    plt.xlabel("Number of MPI Ranks", fontsize=12)
    plt.ylabel("Execution Time (seconds)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    time_file = f"{output_prefix}_time.png"
    plt.savefig(time_file, dpi=300)
    print(f"✓ Saved: {time_file}")
    plt.close()
    
    # --- Plot 2: Weak Scaling Efficiency ---
    plt.figure(figsize=(10, 6))
    plt.plot(df["ranks"], df["scaled_efficiency"], marker='s', linewidth=2, markersize=8, color='purple')
    plt.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Perfect Weak Scaling')
    plt.axhline(y=0.8, color='orange', linestyle=':', linewidth=1.5, label='80% Threshold')
    plt.title(f"Weak Scaling Efficiency: {title}", fontsize=14, fontweight='bold')
    plt.xlabel("Number of MPI Ranks", fontsize=12)
    plt.ylabel("Scaled Efficiency", fontsize=12)
    plt.ylim(0, 1.2)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    efficiency_file = f"{output_prefix}_weak_efficiency.png"
    plt.savefig(efficiency_file, dpi=300)
    print(f"✓ Saved: {efficiency_file}")
    plt.close()
    
    # Print summary
    print(f"\n{title} - Weak Scaling Summary:")
    print(f"{'Ranks':<8} {'Samples/Rank':<15} {'Total Samples':<15} {'Time (s)':<12} {'Efficiency':<12}")
    print("-" * 70)
    for _, row in df.iterrows():
        samples_per = row.get('samples_per_rank', row['total_samples'] // row['ranks'])
        print(f"{int(row['ranks']):<8} {int(samples_per):<15} {int(row['total_samples']):<15} {row['time_sec']:<12.4f} {row['scaled_efficiency']:<12.2%}")
    print()

def main():
    parser = argparse.ArgumentParser(description='Plot Monte Carlo scaling results')
    parser.add_argument('--input', type=str, help='Input CSV file (if not provided, searches results/)')
    parser.add_argument('--output', type=str, help='Output file prefix (without extension)')
    parser.add_argument('--weak', action='store_true', help='Plot weak scaling instead of strong scaling')
    args = parser.parse_args()
    
    # If no input specified, try to find CSV files in results/
    if args.input:
        files_to_plot = [args.input]
    else:
        # Look for CSV files in results directory
        pi_files = [f for f in os.listdir('results') if f.startswith('scaling_data') and f.endswith('.csv')]
        opt_files = [f for f in os.listdir('results') if f.startswith('options_data') and f.endswith('.csv')]
        
        if not pi_files and not opt_files:
            print("Error: No CSV files found in results/. Run experiments first or specify --input.")
            sys.exit(1)
        
        files_to_plot = []
        if pi_files:
            files_to_plot.append(('results/' + pi_files[0], 'Monte Carlo Pi'))
        if opt_files:
            files_to_plot.append(('results/' + opt_files[0], 'Black-Scholes Options'))
    
    # Process each file
    for file_info in files_to_plot:
        if isinstance(file_info, tuple):
            filename, title = file_info
        else:
            filename = file_info
            title = os.path.basename(filename).replace('.csv', '').replace('_', ' ').title()
        
        if not os.path.exists(filename):
            print(f"Warning: {filename} not found. Skipping.")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing: {filename}")
        print(f"{'='*60}")
        
        # Read CSV
        try:
            df = pd.read_csv(filename)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue
        
        # Determine output prefix
        if args.output:
            output_prefix = args.output
        else:
            output_prefix = filename.replace('.csv', '')
        
        # Plot based on mode
        if args.weak:
            plot_weak_scaling(df, title, output_prefix)
        else:
            plot_strong_scaling(df, title, output_prefix)

if __name__ == "__main__":
    main()
