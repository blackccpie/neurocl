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
project (raspicam)

include_directories(
${CMAKE_CURRENT_SOURCE_DIR}
${CMAKE_CURRENT_SOURCE_DIR}/dependencies
${CMAKE_CURRENT_SOURCE_DIR}/dependencies/mmal
${CMAKE_CURRENT_SOURCE_DIR}/dependencies/vcos
)

if (NOT( ${CMAKE_SYSTEM_PROCESSOR} MATCHES "^arm" ) ) #in a pc, adds fake dependencies to mmal functions to enable compilation
set(extra_source private/fake_mmal_dependencies.cpp)
endif ()

set (sources_list
raspicam.cpp
private/private_impl.cpp
private/threadcondition.cpp
${extra_source}
)

set (headers_list
raspicam.h
raspicamtypes.h
private/private_impl.h
private/private_types.h
private/threadcondition.h
private/exceptions.h
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -Wl,--no-as-needed")

add_library(raspicam SHARED ${sources_list} ${headers_list})

if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "^arm" )
find_library(mmalcore_LIBS NAMES mmal_core PATHS /opt/vc/lib)
find_library(mmalutil_LIBS NAMES mmal_util PATHS /opt/vc/lib)
find_library(mmal_LIBS NAMES mmal PATHS /opt/vc/lib)
if ( (NOT mmal_LIBS ) OR (NOT mmalutil_LIBS) OR (NOT mmalcore_LIBS) )
    message(FATAL_ERROR "Could not find mmal libraries")
 endif ()
 set(required_libraries ${mmalcore_LIBS} ${mmalutil_LIBS} ${mmal_LIBS}  )
endif ()

target_link_libraries(raspicam
${required_libraries}
)
