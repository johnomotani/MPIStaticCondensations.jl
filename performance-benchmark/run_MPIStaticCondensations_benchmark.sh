#!/bin/bash

set -e

for np in $(seq $1 -1 1); do
  PROC_PER_CACHE=$(( ($np + 1) / 2 ))
  env MPITRAMPOLINE_MPIEXEC=$PWD/mpiwrapper/bin/mpiwrapper-mpiexec MPITRAMPOLINE_LIB=$PWD/mpiwrapper/lib/libmpiwrapper.so mpirun -np $np julia --project -O3 -t1 benchmark-MPIStaticCondensations.jl
  #env MPITRAMPOLINE_MPIEXEC=$PWD/mpiwrapper/bin/mpiwrapper-mpiexec MPITRAMPOLINE_LIB=$PWD/mpiwrapper/lib/libmpiwrapper.so mpirun --map-by core --bind-to hwthread -np $np julia --project -O3 -t1 benchmark-MPIStaticCondensations.jl
  #env MPITRAMPOLINE_MPIEXEC=$PWD/mpiwrapper/bin/mpiwrapper-mpiexec MPITRAMPOLINE_LIB=$PWD/mpiwrapper/lib/libmpiwrapper.so mpirun --report-bindings --map-by ppr:$PROC_PER_CACHE:L3cache --bind-to core -np $np julia --project -O3 -t1 benchmark-MPIStaticCondensations.jl
done
