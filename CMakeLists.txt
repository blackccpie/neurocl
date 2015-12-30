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

cmake_minimum_required (VERSION 2.8)
project (neurocl)

include_directories(. "../vexcl")

include("../vexcl/cmake/opencl/FindOpenCL.cmake")

set (sources_list
network_manager.cpp
network.cpp
network_bnu.cpp
main.cpp
)

set (headers_list
network_exception.h
network_manager.h
network_interface.h
network.h
network_bnu.h
)

add_executable(neurocl ${sources_list} ${headers_list})

target_link_libraries(neurocl
boost_thread-mt
boost_system-mt
boost_chrono-mt
${OpenCL_LIBRARIES}
)

target_compile_features(neurocl PRIVATE cxx_auto_type)
