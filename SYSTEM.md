# System Configuration

## Cluster
- **Name:** Magic Castle
- **Scheduler:** Slurm

## Nodes Used
| Property | Value |
|----------|-------|
| Node Type | node1-8 |
| CPU | 2 cores per node |
| Memory/Node | 4000 MB per node |

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