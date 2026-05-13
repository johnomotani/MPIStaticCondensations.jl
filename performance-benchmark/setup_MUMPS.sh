#!/bin/bash

# Setup for 'MUMPS' is actually setup for MPItrampoline, so that all the
# Julia-installed binaries (including MUMPS) can use the system MPI.
TOP_LEVEL=$PWD
if [ ! -e MPIwrapper ]; then
  git clone https://github.com/eschnett/MPIwrapper
fi
cd MPIwrapper
cmake -S . -B build -DMPIEXEC_EXECUTABLE=mpirun -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$TOP_LEVEL/mpiwrapper
cmake --build build
cmake --install build

cd $TOP_LEVEL
julia --project -O3 -t1 -e 'using MPIPreferences; MPIPreferences.use_jll_binary("MPItrampoline_jll")'
