#!/bin/bash

module load intel
module load impi
module load mkl

export LD_LIBRARY_PATH=/projects/ilsa8974/apps/taco/install/lib:$LD_LIBRARY_PATH

export LD_LIBRARY_PATH=/curc/sw/intel/17.4/compilers_and_libraries_2017.4.196/linux/tbb/lib/intel64/gcc4.7:$LD_LIBRARY_PATH
