#!/bin/bash

set -e

for np in $(seq $1 -1 1); do
  env MPITRAMPOLINE_MPIEXEC=$PWD/mpiwrapper/bin/mpiwrapper-mpiexec MPITRAMPOLINE_LIB=$PWD/mpiwrapper/lib/libmpiwrapper.so mpirun -np $np julia --project -O3 -t1 benchmark-MPIStaticCondensations.jl
done
