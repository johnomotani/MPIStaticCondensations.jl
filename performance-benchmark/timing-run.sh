#!/bin/bash

FULLDATE=$(date -Iminutes -u)
DATE=${FULLDATE::16}
env MPITRAMPOLINE_MPIEXEC=$PWD/mpiwrapper/bin/mpiwrapper-mpiexec MPITRAMPOLINE_LIB=$PWD/mpiwrapper/lib/libmpiwrapper.so mpirun -np $1 julia --project -O3 -t1 timing-MPIStaticCondensations.jl $2 | tee timing-results-$DATE.txt
