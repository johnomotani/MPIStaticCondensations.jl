#!/bin/bash

FULLDATE=$(date -Iminutes -u)
DATE=${FULLDATE::16}
PROC_PER_CACHE=$(( ($1 + 1) / 2 ))
#env MPITRAMPOLINE_MPIEXEC=$PWD/mpiwrapper/bin/mpiwrapper-mpiexec MPITRAMPOLINE_LIB=$PWD/mpiwrapper/lib/libmpiwrapper.so mpirun --map-by core --bind-to hwthread -np $1 julia --project -O3 -t1 timing-MPIStaticCondensations.jl $2 | tee timing-results-$DATE.txt
#env MPITRAMPOLINE_MPIEXEC=$PWD/mpiwrapper/bin/mpiwrapper-mpiexec MPITRAMPOLINE_LIB=$PWD/mpiwrapper/lib/libmpiwrapper.so mpirun --report-bindings --map-by L3cache --bind-to core -np $1 julia --project -O3 -t1 timing-MPIStaticCondensations.jl $2 | tee timing-results-$DATE.txt
env MPITRAMPOLINE_MPIEXEC=$PWD/mpiwrapper/bin/mpiwrapper-mpiexec MPITRAMPOLINE_LIB=$PWD/mpiwrapper/lib/libmpiwrapper.so OMP_NUM_THREADS=1 mpirun --report-bindings --map-by ppr:$PROC_PER_CACHE:L3cache --bind-to core -np $1 julia --project -O3 -t1 timing-MPIStaticCondensations.jl $2 | tee timing-results-$DATE.txt
