#!/bin/bash

julia --project -O3 -t1 benchmark-multivariable-serial-UMFPACK.jl
