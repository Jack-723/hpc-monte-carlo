# High-Performance Monte Carlo Simulations

## 1. Problem Statement
Monte Carlo simulations are a cornerstone of computational physics and quantitative finance, used to model complex systems with significant uncertainty. However, these methods suffer from slow convergence rates, typically proportional to $1/\sqrt{N}$. To achieve high precision results; whether approximating mathematical constants like $\pi$ or pricing complex financial derivatives—simulation sizes must often exceed billions of iterations.

On a serial processor, these workloads are prohibitively slow. For example, pricing a portfolio of options with high confidence can take hours, which is unacceptable in high-frequency trading environments where latency is critical. The computational challenge is to reduce the "time-to-solution" without sacrificing mathematical accuracy.

## 2. Project Objective
This project implements a parallelized Monte Carlo engine using **Python** and **MPI (Message Passing Interface)** to distribute the computational load across multiple nodes on the Magic Castle cluster. We target two distinct use cases:
1.  **$\pi$ Approximation:** A CPU-bound workload to test raw compute scaling.
2.  **European Options Pricing (Black-Scholes):** A finance workload representing real-world stochastic calculus.

## 3. Success Metrics
We define success based on the following HPC performance metrics:
* **Strong Scaling:** Achieve near-linear speedup as we scale from 1 to 16 MPI ranks.
* **Parallel Efficiency:** Maintain >70% efficiency at maximum scale.
* **Reproducibility:** A fully containerized environment (Apptainer) that yields identical results on any cluster.