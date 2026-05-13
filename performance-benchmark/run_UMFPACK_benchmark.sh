#!/bin/bash

julia --project -O3 -t1 benchmark-serial-UMFPACK.jl
