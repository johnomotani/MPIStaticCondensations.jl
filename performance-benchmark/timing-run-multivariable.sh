#!/bin/bash

FULLDATE=$(date -Iminutes -u)
DATE=${FULLDATE::16}
env MPITRAMPOLINE_MPIEXEC=$PWD/mpiwrapper/bin/mpiwrapper-mpiexec MPITRAMPOLINE_LIB=$PWD/mpiwrapper/lib/libmpiwrapper.so mpirun --map-by core --bind-to hwthread -np $1 julia --project -O3 -t1 timing-multivariable-MPIStaticCondensations.jl $2 | tee timing-multivariable-results-$DATE.txt
