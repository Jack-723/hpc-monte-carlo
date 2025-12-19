# Monte Carlo Simulations for Financial Risk Analysis

**Group 2:** Jack, Omar, Kenny, Salmane, Adrian, Leena

---

## Slide 1: Problem & Impact

### Problem:
- Monte Carlo needs billions of samples for accurate option pricing
- Serial execution: 8+ hours for production portfolios
- Financial institutions need results in minutes, not hours

### Impact:
- Enable real-time risk calculations for banks
- Support Basel III regulatory compliance
- Reduce overnight batch jobs to interactive queries

---

## Slide 2: Approach & Prototype

### Solution: MPI-parallelized Monte Carlo engine in Python

**Two Test Cases:**
- π approximation (CPU benchmark)
- European Call Option pricing (Black-Scholes model)

**How it works:**
1. Distribute N samples across P ranks
2. Each rank generates random samples independently
3. Single MPI_Reduce sums results
4. Master computes final answer

**Reproducibility:** Fixed seeds, Apptainer container, version-pinned dependencies

---

## Slide 3: Scaling Results

### Strong Scaling (100M samples, 2 nodes):

| Ranks | Pi Time | Pi Speedup | Options Time | Options Speedup |
|-------|---------|------------|--------------|-----------------|
| 1     | 1.90s   | 1.0×       | 3.40s        | 1.0×            |
| 2     | 0.95s   | 2.0×       | 1.71s        | 2.0×            |
| 4     | 0.54s   | 3.5×       | 0.94s        | 3.6×            |
| 8     | 0.37s   | 5.2×       | 0.58s        | 5.9×            |

### Key Findings:
- **Options pricing: 73% efficiency at 8 ranks**
- **Pi approximation: 64% efficiency at 8 ranks**
- **Communication overhead < 1%**

---

## Slide 4: EuroHPC Resource Request

### Target System: LUMI-C (AMD EPYC 7763, 64 cores/node)

**Objectives:**
- Scale to 512-1024 MPI ranks
- Production portfolio: 1000 options × 10⁹ paths each
- Test mixed-precision (float32/float64) for 2× speedup

**Resource Request: 50,000 CPU node-hours**

**Justification:**
- Strong scaling experiments: 500 node-hours
- Production runs: 2,500 node-hours
- Contingency + parameter sweeps: remainder

---

## Slide 5: Risks, Milestones & Support Needed

### Risks:
- **Memory bandwidth saturation above 256 ranks** → Use hierarchical reduction
- **Float32 precision loss** → Kahan summation for accumulation
- **I/O bottleneck for large outputs** → Use HDF5 parallel I/O

### Milestones:
- **Month 1-2:** Port to LUMI, validate correctness
- **Month 3-4:** Scaling study (1-512 ranks)
- **Month 5:** Mixed-precision optimization
- **Month 6:** Production runs, submit paper

### Support Needed:
- Profiling tools (Intel VTune, Arm Forge)
- Technical guidance on LUMI MPI configuration
- 10 TB scratch storage
