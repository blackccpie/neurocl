#The MIT License
#
#Copyright (c) 2015-2016 Albert Murienne
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in
#all copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#THE SOFTWARE.

cmake_minimum_required (VERSION 3.0)
project (alnet)

include_directories( ../../src )

set (headers_list
layer.h
conv_layer.h
input_layer.h
output_layer.h
pool_layer.h
optimizer.h
tensor.h
lenet.h
lenet_manager.h
lenet_interface.h
lenet_file_handler.h
)

set (sources_list
main.cpp
lenet.cpp
lenet_manager.cpp
lenet_file_handler.cpp
tensor.cpp
)

add_executable(alnet ${sources_list} ${headers_list})

target_link_libraries(alnet
neurocl
)
