# Team Status Update - December 6, 2025

## What's Done ✅

### Code & Infrastructure (Person 1 - Lead)
- ✅ Repository structure complete
- ✅ All Slurm submission scripts created
- ✅ Reproducibility fixes applied (fixed seeds in code)
- ✅ Makefile updated
- ✅ Documentation templates ready

### Local Testing Complete
- ✅ Pi approximation works (1, 2, 4 cores tested)
- ✅ Options pricing works (1, 2, 4 cores tested)
- ✅ Plotting scripts functional
- ✅ Results saved in `results/` folder

## Waiting On 🕐

### Person 2 - Cluster Runs (BLOCKED)
**Status:** Magic Castle cluster is DOWN for maintenance
**When available:**
- Run `sbatch slurm/submit_pi_scaling.sbatch`
- Run `sbatch slurm/submit_options_scaling.sbatch`
- Run `sbatch slurm/weak_scaling.sbatch`
- Collect CSV results
- Update `SYSTEM.md` with actual node specs

**NOTE:** Must edit account name in `.sbatch` files first!
```bash
# Change this line in all .sbatch files:
#SBATCH --account=def-someprof
# To your actual account:
#SBATCH --account=def-[your-prof-id]
```

### Person 3 - Profiling (BLOCKED)
**Depends on:** Cluster coming back online
**Tasks:**
- Run `perf record` on 8-rank job
- Save output to `results/perf_report.txt`
- Get `sacct` logs
- Write bottleneck analysis (1 paragraph)

### Person 4 - Weak Scaling (BLOCKED)
**Depends on:** Cluster coming back online
**Tasks:**
- Run weak scaling experiments
- Plot results (script already updated)
- Try one optimization (e.g., NumPy vectorization)

## Can Do NOW (Cluster Down) 📝

### Person 5 - Paper ✍️
**Status:** Template ready at `docs/paper_template.md`
**All data filled in - just needs conversion to PDF!**

**To do:**
1. Open `docs/paper_template.md`
2. Review content (already filled with local data)
3. Add note: "Full cluster runs pending system availability"
4. Convert to PDF:
   - Option 1: https://www.markdowntopdf.com/
   - Option 2: Pandoc command (if installed)
   - Option 3: Copy to Word/LaTeX
5. Save as `docs/paper.pdf`

### Person 6 - Proposal & Pitch ✍️
**Status:** Templates ready at `docs/eurohpc_proposal_template.md` and `docs/pitch_slides_content.md`
**All data filled in - just needs formatting!**

**To do:**
1. **Proposal:**
   - Open `docs/eurohpc_proposal_template.md`
   - Already complete with calculations
   - Convert to PDF → save as `docs/eurohpc_proposal.pdf`

2. **Pitch Slides:**
   - Open `docs/pitch_slides_content.md`
   - Copy content to PowerPoint/Google Slides
   - Add existing plots from `results/` folder
   - Add note on slides: "Full cluster runs in progress"
   - Export as `docs/pitch_slides.pdf`

## Current Grade Estimate

**If submitted NOW:** ~60-70/100
- ✅ Code works (local validation)
- ✅ Repo structure correct
- ✅ Documentation templates filled
- ⚠️ Missing cluster runs (8-16 ranks)
- ⚠️ Missing profiling data
- ⚠️ Missing weak scaling

**After cluster runs complete:** ~85-95/100

## Timeline

### Today (Cluster Down):
- Person 5: Convert paper to PDF
- Person 6: Create pitch slides in PowerPoint
- Person 6: Convert proposal to PDF

### When Cluster Comes Back:
1. Person 2: Edit account names, submit jobs (~5 min)
2. Person 2: Monitor jobs (~1-2 hours)
3. Person 3: Run profiler on one job (~30 min)
4. Person 4: Run weak scaling (~1 hour)
5. All: Update paper with final numbers (~1 hour)

## Files Ready to Submit

```
✅ src/                  - All code works
✅ slurm/                - Submission scripts ready
✅ env/                  - Environment files ready
✅ data/README.md        - Data documentation
✅ results/              - Local test results + plots
✅ docs/paper_template.md         - FILLED, needs PDF conversion
✅ docs/eurohpc_proposal_template.md  - FILLED, needs PDF conversion  
✅ docs/pitch_slides_content.md   - FILLED, needs PowerPoint
✅ reproduce.md          - Reproducibility guide
✅ SYSTEM.md            - System specs (update after cluster runs)
✅ README.md            - Project overview
```

## Questions?

Ask in group chat or check these files:
- `reproduce.md` - How to run everything
- `SYSTEM.md` - What hardware/software we use
- This file - Current status

---
**Last Updated:** December 6, 2025  
**Status:** Ready for cluster runs + documentation finalization
