#!/bin/bash

./run_UMFPACK_benchmark.sh
./run_MPIStaticCondensations_benchmark.sh $1
./run_MUMPS_benchmark.sh $1
