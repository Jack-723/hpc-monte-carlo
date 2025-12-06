# System Configuration

## Cluster
- **Name:** Magic Castle HPC
- **Scheduler:** Slurm 24.11.6
- **Account:** def-sponsor00
- **Login Node:** login1.hpcie.labs.faculty.ie.edu

## Nodes Used
| Property | Value |
|----------|-------|
| Node Names | gpu-node1, gpu-node2 |
| Architecture | x86_64 |
| CPU per Node | 4 cores (4 sockets × 1 core/socket × 1 thread/core) |
| Memory per Node | 28000 MB (28 GB) |
| GPU | 1 GPU per node |
| OS | Linux 5.14.0-570.41.1.el9_6.x86_64 |
| Partitions | cpubase_bycore_b1, gpu-node |

## Job Configuration
| Property | Value |
|----------|-------|
| Nodes | 2 |
| Tasks per Node | 4 |
| Total MPI Ranks | 8 (max) |
| CPUs per Task | 1 |
| Memory per CPU | 2 GB |

## Software Modules
| Software | Version |
|----------|---------|
| GCC | 12.3 |
| OpenMPI | 4.1.5 |
| Python | 3.11.5 |
| mpi4py | 4.0.3 |
| flexiblas | 3.3.1 |
| UCX | 1.14.1 |
| PMIx | 4.2.4 |

## Python Packages
| Package | Version |
|----------|---------|
| numpy | >=1.21.0 |
| mpi4py | 4.0.3 |
| pandas | >=1.3.0 |
| matplotlib | >=3.4.0 |

## Commands to get system info
```bash
sinfo -N -l
scontrol show node gpu-node1
scontrol show node gpu-node2
module list
python --version
```