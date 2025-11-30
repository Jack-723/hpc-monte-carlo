import argparse
import time
import numpy as np
from mpi4py import MPI

def estimate_pi(local_samples, seed):
    # Use the rank-specific seed to ensure randomness across processors
    np.random.seed(seed)
    
    # Generate random points
    x = np.random.rand(local_samples)
    y = np.random.rand(local_samples)
    
    # Calculate how many fell inside the circle
    inside_circle = np.sum(x**2 + y**2 <= 1.0)
    return inside_circle

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=10000000) # Default 10 million
    args = parser.parse_args()

    # Split work
    total_samples = args.samples
    local_samples = total_samples // size
    # Add remainder to the first few ranks
    if rank < total_samples % size:
        local_samples += 1

    if rank == 0:
        print(f"Running Monte Carlo Pi on {size} processors.")
        print(f"Total Samples: {total_samples}")
        start_time = time.time()

    # --- THE WORK ---
    # We use (rank + time) to ensure every run is random
    my_count = estimate_pi(local_samples, seed=rank + int(time.time()))
    
    # --- THE COMMUNICATION ---
    # Sum up all 'my_count' values into 'total_count' on Rank 0
    total_count = comm.reduce(my_count, op=MPI.SUM, root=0)

    if rank == 0:
        end_time = time.time()
        elapsed = end_time - start_time
        pi_est = (4.0 * total_count) / total_samples
        
        print(f"Pi Estimate: {pi_est}")
        print(f"Error:       {abs(pi_est - np.pi)}")
        print(f"Time:        {elapsed:.4f} sec")
        
        # Save to CSV for the assignment graphs
        with open("results/scaling_data.csv", "a") as f:
            f.write(f"{size},{total_samples},{elapsed},{pi_est}\n")

if __name__ == "__main__":
    main()