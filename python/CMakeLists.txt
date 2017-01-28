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

cmake_minimum_required (VERSION 3.2)
project (pyneurocl)

include_directories(
    "${CMAKE_SOURCE_DIR}/../CImg"
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/utils"
)

# Find necessary python packages
find_package( PythonLibs 2.7 REQUIRED )
include_directories(
    ${PYTHON_INCLUDE_DIRS}
)

if (APPLE)
    include_directories(
        "/System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python/numpy/core/include"
    )
endif ()

set (sources_list
pyneurocl.cpp
)

if (NOT APPLE)
	add_definitions(-Wno-cpp)
else ()
	add_definitions("-Wno-#warnings")
endif ()

add_library(pyneurocl SHARED ${sources_list} ${headers_list})

set_target_properties(pyneurocl PROPERTIES SUFFIX .pyd)

target_link_libraries(pyneurocl
boost_python${boost_suffix}
python2.7
imagetools
neurocl
)

# don't prepend wrapper library name with lib
set_target_properties( pyneurocl PROPERTIES PREFIX "" )
