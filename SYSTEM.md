# System Configuration

## Cluster
- **Name:** Magic Castle
- **Scheduler:** Slurm

## Nodes Used
| Property | Value |
|----------|-------|
| Node Type | TBD |
| CPU | TBD |
| Cores/Node | TBD |
| Memory/Node | TBD |

*Update after running `scontrol show node` on the cluster*

## Software
| Software | Version |
|----------|---------|
| GCC | 11.3.0 |
| OpenMPI | 4.1.4 |
| Python | 3.10 |
| numpy | >=1.21.0 |
| mpi4py | >=3.1.0 |

## Commands to get system info
```bash
sinfo -N -l
scontrol show node <nodename>
module list
python --version
```