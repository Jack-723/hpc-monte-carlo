# Data Directory

This Monte Carlo simulation is **compute-bound** and does not require external input datasets.

## Pi Approximation
- Generates random (x, y) points uniformly distributed in [0, 1]
- No input data required

## Options Pricing (Black-Scholes)
- Generates random paths using Geometric Brownian Motion
- Parameters are hardcoded in the script:
  - S0 = 100.0 (Initial stock price)
  - K = 105.0 (Strike price)
  - T = 1.0 (Time to maturity)
  - r = 0.05 (Risk-free rate)
  - σ = 0.2 (Volatility)

## Reproducibility

Use the `--seed` flag for reproducible results:
```bash
mpiexec -n 4 python src/monte_carlo.py --samples 10000000 --seed 42
```
```

---

### File 2: `env/modules.txt`

Open `env/modules.txt` and paste this:
```
# Modules for Magic Castle cluster
# Run: module load <module_name>

gcc/11.3.0
openmpi/4.1.4
python/3.10

# Note: Run 'module avail' on your cluster to check available versions