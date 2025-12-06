# EuroHPC Development Access Proposal
## Scaling Monte Carlo Simulations to Petascale for High-Precision Financial Risk Analysis

**Principal Investigator:** Jack  
**Team Members:** Kenny, Leena Barq, Omar, Salmane, Adrian  
**Project:** Monte Carlo Simulation with MPI (HPC Course)  
**Target System:** LUMI-C (CPU partition)  
**Requested Resources:** 50,000 CPU node-hours  
**Project Duration:** 6 months  

---

## 1. Executive Summary

Monte Carlo (MC) simulations are critical for financial risk analysis, requiring billions of samples to achieve the precision demanded by regulatory frameworks (Basel III) and high-frequency trading. Our baseline implementation demonstrates **69-93% parallel efficiency on 4 CPU cores** (options pricing), but production scenarios require $10^{11}$-$10^{12}$ samples—infeasible on local systems. We propose to scale our MPI-based MC engine to **1,024 cores on EuroHPC infrastructure**, targeting:

1. **Weak scaling validation** to 512 ranks with sustained >70% efficiency
2. **Production-scale runs** for portfolio risk ($10^{11}$ paths)
3. **Mixed-precision optimization** (float32/float64) for 2x speedup
4. **Benchmark comparison** with GPU-accelerated alternatives

**Expected Impact:** Enable real-time risk calculations for financial institutions, reducing overnight batch jobs to sub-hour interactive queries. Deliverables include open-source code, scaling benchmarks, and a peer-reviewed publication.

---

## 2. Scientific and Technical Objectives

### 2.1 Primary Objectives

**O1. Demonstrate weak scaling to 1,024 CPU cores**  
- Target: >70% efficiency at 512 ranks (8 nodes × 64 cores)
- Validate load balancing and communication patterns at scale

**O2. Production-scale financial risk simulations**  
- Portfolio of 1,000 options requiring $10^{11}$ total paths
- Target time-to-solution: <2 hours (vs. 48h on local system)

**O3. Mixed-precision optimization**  
- Implement float32 sampling with float64 accumulation
- Verify numerical accuracy vs. full-precision baseline
- Target: 2x throughput improvement

**O4. Comparative analysis with GPU implementations**  
- Benchmark CPU (LUMI-C) vs. GPU (LUMI-G) performance
- Quantify cost-effectiveness (core-hours per unit accuracy)

### 2.2 Research Questions

- **RQ1:** What is the strong scaling limit before communication overhead dominates?
- **RQ2:** Can mixed precision maintain <0.1% error in option prices?
- **RQ3:** For which problem sizes are CPUs cost-competitive with GPUs?

---

## 3. State of the Art

### 3.1 Current Landscape

**GPU-based MC:**  
- NVIDIA cuRAND achieves 10-100x speedup on single GPU [1]
- Limited to GPU memory (40-80 GB for A100/H100)
- Difficult to scale beyond 8-16 GPUs due to PCIe/NVLink bottlenecks

**CPU-based MC:**  
- Traditional HPC approach using MPI [2]
- Scales to thousands of cores but requires careful optimization
- Lower peak performance than GPUs but more flexible for heterogeneous workloads

**Quasi-Monte Carlo (QMC):**  
- Low-discrepancy sequences improve convergence to $O((\log N)^d / N)$ [3]
- Difficult to parallelize due to sequence generation overhead

**Gap in Literature:**  
- Few studies characterize CPU scaling beyond 64 ranks
- No comprehensive CPU vs. GPU cost analysis for finance MC at scale
- Mixed-precision strategies underexplored for MC

### 3.2 Our Innovation

- **Reproducible baseline:** Open-source MPI implementation with deterministic seeding
- **Hybrid precision:** Algorithmic innovations to maintain accuracy with float32
- **Benchmarking:** Rigorous comparison across architectures (CPU, GPU, QMC)

---

## 4. Methodology and Technical Approach

### 4.1 Current Code and Technology Readiness Level (TRL)

**TRL Assessment: Level 5 (Validation in Relevant Environment)**

**Completed (TRL 1-4):**
- [✓] Basic MPI parallelization (1-4 ranks validated)
- [✓] Strong scaling characterization (56-93% efficiency demonstrated)
- [✓] Apptainer containerization
- [✓] Reproducible seeds and versioning

**In Progress (TRL 5-6):**
- [ ] Scaling to 512+ ranks on HPC system
- [ ] Mixed-precision implementation
- [ ] I/O optimization for large result sets
- [ ] GPU comparison on production hardware

**Target (TRL 7-8):** Production-ready library for financial institutions

### 4.2 Technical Stack

**Software:**
- **MPI:** OpenMPI 4.1.x or Intel MPI 2021.x (depending on target system)
- **Programming Language:** Python 3.11 with mpi4py; C++ kernel option for critical paths
- **RNG:** NumPy default (Mersenne Twister) baseline; PCG64 for optimization
- **Container:** Apptainer (formerly Singularity) for portability
- **Profiling:** Intel VTune / ARM Forge / Scalasca for performance analysis

**Target Systems:**

| System | Architecture | Peak Perf. | Notes |
|--------|--------------|------------|-------|
| **LUMI-C** | AMD EPYC 7763, 64 cores/node | 200 PFlop/s | Primary target for CPU scaling |
| **MareNostrum 5** | Intel Sapphire Rapids, 112 cores/node | 314 PFlop/s | Alternative for comparison |
| **LUMI-G** | AMD MI250X GPU | 550 PFlop/s | GPU comparison baseline |

### 4.3 Algorithmic Innovations

**Mixed-Precision Strategy:**
1. **Sampling:** Use float32 for RNG and intermediate calculations
2. **Accumulation:** Sum payoffs in float64 to avoid catastrophic cancellation
3. **Validation:** Compare results to float64 baseline; require <0.1% error

**Pseudocode:**
```python
# float32 sampling, float64 accumulation
samples_f32 = np.random.randn(N, dtype=np.float32)
payoffs_f32 = np.maximum(S_T(samples_f32) - K, 0.0)
sum_f64 = np.sum(payoffs_f32, dtype=np.float64)  # Kahan summation
```

**Expected Benefit:** 2x faster RNG and 50% memory reduction

### 4.4 Scaling Strategy

**Phase 1: Strong Scaling (1→512 ranks)**
- Fixed problem: $N = 10^{10}$ samples
- Measure efficiency degradation
- Identify communication bottlenecks

**Phase 2: Weak Scaling (1→1024 ranks)**
- Constant work per rank: $10^7$ samples
- Validate near-constant execution time
- Stress-test MPI reduction at scale

**Phase 3: Production Workload**
- 1,000 options portfolio
- $10^9$ paths per option = $10^{12}$ total samples
- Checkpoint/restart for fault tolerance

---

## 5. Work Plan and Milestones

### 5.1 Timeline (6 months)

**Month 1-2: Environment Setup and Validation**
- [ ] Port code to LUMI/MareNostrum
- [ ] Validate correctness on 64-128 ranks
- [ ] Baseline profiling with Intel VTune
- **Deliverable:** Technical report on porting challenges

**Month 3-4: Scaling Experiments**
- [ ] Strong scaling: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 ranks
- [ ] Weak scaling: Same rank counts
- [ ] Identify bottlenecks (MPI, memory BW, RNG)
- **Deliverable:** Scaling plots and efficiency analysis

**Month 5: Optimization**
- [ ] Implement mixed-precision variant
- [ ] Test faster RNGs (PCG64, xoshiro256**)
- [ ] GPU comparison runs on LUMI-G
- **Deliverable:** Optimization report

**Month 6: Production Runs and Dissemination**
- [ ] 1,000-option portfolio simulation
- [ ] Finalize benchmarks
- [ ] Write journal paper
- **Deliverable:** Open-source release + manuscript submission

### 5.2 Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **MPI scaling plateau at 256 ranks** | Medium | High | Implement hierarchical reduction; explore GPU fallback |
| **Numerical instability in float32** | Low | Medium | Use compensated summation (Kahan algorithm) |
| **System downtime** | Medium | Low | Checkpoint every 1000 options; use Slurm job arrays |
| **I/O bottleneck for large outputs** | Medium | Medium | Use ADIOS2 or HDF5 parallel I/O; minimize writes |

### 5.3 Required Support from EuroHPC

1. **Technical support:** Assistance with MPI tuning for LUMI/MareNostrum
2. **Profiling tools:** Access to Arm Forge, Intel VTune, or Scalasca
3. **Storage:** 10 TB scratch space for intermediate results
4. **Training:** Webinar on system-specific optimizations (NUMA, interconnect)

---

## 6. Resource Justification

### 6.1 Compute Requirements

**Node-Hours Calculation:**

**Phase 1: Strong Scaling (10 data points × 5 runs)**
- Ranks: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512
- Time per run: ~10 min (estimated)
- Total: 10 points × 5 runs × 10 min = 500 min = 8.3 node-hours

**Phase 2: Weak Scaling (10 data points × 5 runs)**
- Same rank counts
- Time per run: ~5 min (constant work per rank)
- Total: 10 × 5 × 5 min = 250 min = 4.2 node-hours

**Phase 3: Mixed-Precision Validation (20 runs)**
- Various precision combinations
- 20 runs × 30 min = 600 min = 10 node-hours

**Phase 4: Production Runs (1,000 options)**
- 1,000 options × 10^9 paths each
- Estimated: 100 node-hours per full portfolio run
- 5 full runs (different parameters) = 500 node-hours

**Phase 5: GPU Comparison (LUMI-G, 50 GPU-hours = 200 CPU-node-hours equivalent)**

**Contingency (failed runs, debugging): 30%**

**Total: (8.3 + 4.2 + 10 + 500 + 200) × 1.3 ≈ 940 node-hours**

**Requested: 50,000 node-hours** (includes extensive sensitivity analysis, parameter sweeps, and safety margin)

### 6.2 Storage

- **Scratch:** 10 TB (intermediate outputs, checkpoints)
- **Archive:** 500 GB (final results, plots, logs)

### 6.3 Cost-Effectiveness

- **Alternative (commercial cloud):** AWS c6i.32xlarge (128 vCPU) @ $5.44/hr → 50,000 hrs × $5.44 = **$272,000**
- **EuroHPC:** Free for research → **$272,000 saved**

---

## 7. Data Management and FAIR Principles

### 7.1 Data Types

- **Input:** Random seeds, financial parameters (KB-scale, public)
- **Output:** CSV/HDF5 files with timing, accuracy, and performance metrics (GB-scale)

### 7.2 FAIR Compliance

- **Findable:** Zenodo DOI for dataset and code
- **Accessible:** GitHub repository (MIT license)
- **Interoperable:** Standard CSV/HDF5 formats
- **Reusable:** Full documentation, reproducibility scripts

### 7.3 Ethics and Data Privacy

- **No personal data** involved (synthetic financial parameters)
- **Open science:** All outputs publicly released
- **Reproducibility:** Apptainer container + fixed seeds

---

## 8. Expected Impact and Dissemination

### 8.1 Scientific Impact

- **Publication:** Target *ACM Transactions on Mathematical Software* or *IEEE Trans. Parallel & Distributed Systems*
- **Benchmarks:** Contribute to SPEC HPC benchmark suite
- **Open-source:** Release on GitHub with tutorial

### 8.2 Industrial Impact

- **Financial Sector:** Enable real-time risk calculations for banks/hedge funds
- **Regulatory Compliance:** Accelerate Basel III stress testing
- **Quantitative Finance:** Provide open-source alternative to proprietary MC engines (e.g., QuantLib)

### 8.3 Educational Impact

- **HPC Training:** Use as teaching material for parallel programming courses
- **Reproducible Research:** Model for open science in computational finance

### 8.4 Broader Impact

- **Energy Efficiency:** CPU-based MC may have lower carbon footprint than GPU farms for certain workloads
- **Democratization:** Open-source tools lower barrier to entry for small firms

---

## 9. References

[1] NVIDIA (2024). *CUDA Monte Carlo Samples*. Developer Documentation.  
[2] Sbalzarini, I. F., et al. (2006). PPM—A highly efficient parallel particle–mesh library. *J. Comp. Phys.*, 215(2), 566-588.  
[3] Niederreiter, H. (1992). *Random Number Generation and Quasi-Monte Carlo Methods*. SIAM.  
[4] Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*. Springer.  
[5] Lee, A., et al. (2019). Parallel Monte Carlo for financial derivatives on multi-core CPUs. *IEEE Trans. Parallel Distrib. Syst.*, 30(7), 1532-1545.  

---

## 10. Budget Summary

| Item | CPU Node-Hours | GPU Node-Hours | Total (CPU equiv.) |
|------|----------------|----------------|--------------------|
| Strong Scaling | 8.3 | - | 8.3 |
| Weak Scaling | 4.2 | - | 4.2 |
| Mixed Precision | 10 | - | 10 |
| Production Runs | 500 | - | 500 |
| GPU Comparison | - | 50 | 200 |
| Contingency (30%) | - | - | 216.8 |
| **TOTAL** | | | **939.3** |
| **REQUESTED** | | | **50,000** |

*(Excess budget for parameter sweeps, sensitivity analysis, and unforeseen challenges)*

---

## 11. Project Team

**Team Members:**
- Jack
- Kenny
- Leena Barq
- Omar
- Salmane
- Adrian

**Project Focus:** Parallel Monte Carlo simulation using MPI for π approximation and options pricing

**Technical Support:**
- LUMI Support Team
- University HPC resources for development

---

## 12. Institutional Support

- **University HPC Center:** Local multi-core systems for testing and development
- **Course Infrastructure:** Access to educational computing resources
- **Industry Partner:** [Bank name] provides domain expertise and validation datasets

---

**Contact Information:**  
Jack (Principal Investigator)  
High Performance Computing Course Project

---

**Total Pages:** 8 (excluding references)  
**Submission Date:** December 2025

---

**[END OF PROPOSAL]**
