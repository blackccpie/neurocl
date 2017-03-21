#!/bin/sh

mkdir -p build_gcc

cd build_gcc
cmake -DCMAKE_CXX_FLAGS="-march=armv6 -mfloat-abi=hard -mfpu=vfp" -DNEUROCL_DISABLE_VEXCL=1 -DCMAKE_BUILD_TYPE=Release ..
make
