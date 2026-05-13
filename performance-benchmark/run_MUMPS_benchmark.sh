#!/bin/bash

set -e

NPROCS=$1

for np in $(seq $NPROCS -1 1); do for nt in $(seq $NPROCS -1 1); do
  if julia -e "if ($np * $nt > $NPROCS) exit(1) else exit(0) end"; then
    env MPITRAMPOLINE_MPIEXEC=$PWD/mpiwrapper/bin/mpiwrapper-mpiexec MPITRAMPOLINE_LIB=$PWD/mpiwrapper/lib/libmpiwrapper.so OMP_NUM_THREADS=$nt mpirun -np $np julia --project -O3 -t$nt benchmark-MUMPS.jl
  fi
done; done
