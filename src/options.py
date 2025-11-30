import argparse
import time
import numpy as np
from mpi4py import MPI

def simulate_paths(S0, K, T, r, sigma, samples, seed):
    """
    S0: Initial stock price
    K:  Strike price
    T:  Time to maturity (years)
    r:  Risk-free interest rate
    sigma: Volatility
    """
    np.random.seed(seed)
    
    # Generate random paths (geometric brownian motion)
    z = np.random.standard_normal(samples)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    
    # Calculate payoff for Call Option: max(ST - K, 0)
    payoffs = np.maximum(ST - K, 0.0)
    
    return np.sum(payoffs)

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Parameters for the Option (Apple Stock-ish)
    S0 = 100.0   # Current Price
    K = 105.0    # Strike Price
    T = 1.0      # One year
    r = 0.05     # 5% interest
    sigma = 0.2  # 20% volatility
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=10000000)
    args = parser.parse_args()

    total_samples = args.samples
    local_samples = total_samples // size
    if rank < total_samples % size:
        local_samples += 1

    if rank == 0:
        print(f"Running Options Pricing (Black-Scholes) on {size} processors.")
        start_time = time.time()

    # --- THE WORK ---
    local_sum = simulate_paths(S0, K, T, r, sigma, local_samples, seed=rank + int(time.time()))

    # --- THE COMMUNICATION ---
    # Sum all payoffs from everyone
    global_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)

    if rank == 0:
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Average the payoff and discount back to present value
        average_payoff = global_sum / total_samples
        option_price = np.exp(-r * T) * average_payoff
        
        print(f"Option Price: ${option_price:.4f}")
        print(f"Time:         {elapsed:.4f} sec")
        
        # Append to a new results file
        with open("results/options_data.csv", "a") as f:
            f.write(f"{size},{total_samples},{elapsed},{option_price}\n")

if __name__ == "__main__":
    main()