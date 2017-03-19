#!/bin/sh

mkdir -p build_gcc

cd build_gcc
cmake -DCMAKE_CXX_FLAGS="-march=armv8-a" -DNEUROCL_DISABLE_VEXCL=0 -DCMAKE_BUILD_TYPE=Release ..
make -j5
