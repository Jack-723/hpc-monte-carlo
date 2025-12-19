# High-Performance Monte Carlo Simulations

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![MPI](https://img.shields.io/badge/MPI-OpenMPI%204.1.5-green.svg)](https://www.open-mpi.org/)

## 1. Problem Statement

Monte Carlo simulations are a cornerstone of computational physics and quantitative finance, used to model complex systems with significant uncertainty. However, these methods suffer from slow convergence rates, typically proportional to $1/\sqrt{N}$. To achieve high precision results—whether approximating mathematical constants like $\pi$ or pricing complex financial derivatives—simulation sizes must often exceed billions of iterations.

On a serial processor, these workloads are prohibitively slow. For example, pricing a portfolio of options with high confidence can take hours, which is unacceptable in high-frequency trading environments where latency is critical. The computational challenge is to reduce the "time-to-solution" without sacrificing mathematical accuracy.

## 2. Project Objective

This project implements a parallelized Monte Carlo engine using **Python** and **MPI (Message Passing Interface)** to distribute the computational load across multiple nodes on the Magic Castle cluster. We target two distinct use cases:

1. **$\pi$ Approximation:** A CPU-bound workload to test raw compute scaling.
2. **European Options Pricing (Black-Scholes):** A finance workload representing real-world stochastic calculus.

## 3. Success Metrics

We define success based on the following HPC performance metrics:

* **Strong Scaling:** Achieve near-linear speedup as we scale from 1 to 8 MPI ranks.
* **Parallel Efficiency:** Maintain >70% efficiency at maximum scale.
* **Reproducibility:** A fully containerized environment (Apptainer) that yields identical results on any cluster.

**Achieved:** 5.86× speedup at 8 ranks (73% efficiency) for options pricing.

## 4. Quick Start

### Prerequisites
- Magic Castle cluster access (or any HPC system with Slurm)
- OpenMPI 4.1.5+
- Python 3.11+

### Installation

```bash
# Clone repository
git clone https://github.com/Jack-723/hpc-monte-carlo.git
cd hpc-monte-carlo

# Load modules (on Magic Castle)
source env/load_modules.sh

# Install Python dependencies
pip install -r requirements.txt
```

### Local Testing

```bash
# Run Pi approximation with 4 MPI ranks
mpirun -n 4 python src/monte_carlo.py --samples 10000000 --seed 42

# Run options pricing
mpirun -n 4 python src/options.py --samples 10000000 --seed 42

# Generate plots
python src/plot_results.py
```

### Cluster Runs (Slurm)

**IMPORTANT:** Edit `slurm/*.sbatch` files and change `--account=def-someprof` to your actual allocation!

```bash
# Submit strong scaling jobs
sbatch slurm/submit_pi_scaling.sbatch
sbatch slurm/submit_options_scaling.sbatch

# Submit weak scaling job
sbatch slurm/weak_scaling.sbatch

# Check job status
squeue -u $USER

# View results
ls results/
```

## 5. Repository Structure

```
├── src/                  # Source code
│   ├── monte_carlo.py    # Pi approximation
│   ├── options.py        # Options pricing
│   └── plot_results.py   # Visualization
├── slurm/                # Slurm job scripts
│   ├── submit_pi_scaling.sbatch
│   ├── submit_options_scaling.sbatch
│   ├── weak_scaling.sbatch
│   └── container_run.sh
├── env/                  # Environment setup
│   ├── project.def       # Apptainer definition
│   ├── modules.txt       # Module versions
│   └── load_modules.sh   # Module loading script
├── data/                 # Data information (no external datasets needed)
├── results/              # Output CSV and plots
├── docs/                 # Documentation
│   ├── paper.md          # Technical paper (6 pages)
│   └── eurohpc_proposal.md  # EuroHPC proposal (8 pages)
├── reproduce.md          # Reproducibility guide
├── SYSTEM.md            # Hardware/software specs
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## 6. Features

- ✅ **Reproducible:** Fixed seeds ensure identical results across runs
- ✅ **Portable:** Apptainer container for any HPC system
- ✅ **Scalable:** Tested 1-8 MPI ranks with 73% efficiency at max scale
- ✅ **Well-documented:** Complete paper and EuroHPC proposal
- ✅ **Open-source:** MIT license

## 7. Results Summary

### Strong Scaling (100M samples)

| Ranks | Pi Time (s) | Options Time (s) | Speedup | Efficiency |
|-------|-------------|------------------|---------|------------|
| 1     | 1.90        | 3.40             | 1.00×   | 100%       |
| 2     | 0.95        | 1.71             | 2.00×   | 99%        |
| 4     | 0.54        | 0.94             | 3.51×   | 90%        |
| 8     | 0.37        | 0.58             | 5.15×   | 73%        |

### Weak Scaling (10M samples/rank)
## 8. Documentation

- **Reproducibility:** See [reproduce.md](reproduce.md)
- **System specs:** See [SYSTEM.md](SYSTEM.md)
- **Paper:** See [docs/paper.md](docs/paper.md)
- **EuroHPC Proposal:** See [docs/eurohpc_proposal.md](docs/eurohpc_proposal.md)
- **Paper:** See [docs/paper_template.md](docs/paper_template.md)
- **Proposal:** See [docs/eurohpc_proposal_template.md](docs/eurohpc_proposal_template.md)
- **Pitch:** See [docs/pitch_slides_content.md](docs/pitch_slides_content.md)

## 9. Citation

If you use this code in your research, please cite:

```bibtex
@software{hpc_monte_carlo_2025,
```bibtex
@software{hpc_monte_carlo_2025,
  author = {Jack and Kenny and Leena Barq and Omar and Salmane and Adrian},
  title = {High-Performance Monte Carlo Simulations with MPI},
  year = {2025},
  url = {https://github.com/Jack-723/hpc-monte-carlo}
}
```10. License

MIT License - see LICENSE file for details.

## 11. Contact

## 11. Authors

Jack, Kenny, Leena Barq, Omar, Salmane, Adrian

## 12. Acknowledgments (Alliance/EESSI)
- Course instructors and TAs
- OpenMPI and NumPy communities