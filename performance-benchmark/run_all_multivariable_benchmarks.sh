#!/bin/bash

./run_UMFPACK_multivariable_benchmark.sh
./run_MPIStaticCondensations_multivariable_benchmark.sh $1
./run_MUMPS_multivariable_benchmark.sh $1
