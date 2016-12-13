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
project (neurocl)

# check boost version
find_package(Boost 1.58.0 REQUIRED)

# manage 1.58-legacy portable binary archive implementation
# 1.58+ non-legacy implementation should work up to boost 1.62
if (${Boost_MINOR_VERSION} STREQUAL "58")
    # Boost version == 1.58
    add_definitions(-DPORTABLE_BINARY_ARCHIVE_LEGACY)
endif()

# detect SIMD features
if ( "${CMAKE_CXX_FLAGS}" MATCHES "^.*(sse|neon).*$" )

    message("SIMD is enabled for this build")

    add_definitions(-DSIMD_ENABLED)

    set( sources_list_simd
        mlp/network_bnu_fast.cpp
    )

    set (headers_list_simd
        mlp/network_bnu_fast.h
    )
endif ()

include_directories(
    .
    "${CMAKE_SOURCE_DIR}/../vexcl"
    "${CMAKE_SOURCE_DIR}/../CImg"
)

include("${CMAKE_SOURCE_DIR}/../vexcl/cmake/opencl/FindOpenCL.cmake")

set (sources_list
common/samples_manager.cpp
common/learning_scheduler.cpp
common/network_factory.cpp
common/network_manager.cpp
common/logger.cpp

common/portable_binary_archive/portable_binary_iarchive.cpp
common/portable_binary_archive/portable_binary_oarchive.cpp

mlp/network_file_handler.cpp
mlp/network_vexcl.cpp
mlp/network_bnu_base.cpp
mlp/network_bnu_ref.cpp

convnet/network.cpp
convnet/network_parallel.cpp
convnet/network_file_handler.cpp
convnet/tensor.cpp
convnet/tensor_operations.cpp
)

set (headers_list

neurocl.h

interfaces/network_interface.h
interfaces/network_manager_interface.h
interfaces/network_file_handler_interface.h

common/samples_manager.h
common/learning_scheduler.h
common/network_factory.h
common/network_manager.h
common/network_sample.h
common/network_exception.h
common/network_utils.h
common/network_config.h
common/iterative_trainer.h
common/logger.h
common/solver.h
common/thread_pool.h

common/portable_binary_archive/portable_binary_archive.hpp
common/portable_binary_archive/portable_binary_iarchive.hpp
common/portable_binary_archive/portable_binary_oarchive.hpp

mlp/network_file_handler.h
mlp/network_manager_mlp.h
mlp/network_interface_mlp.h
mlp/network_vexcl.h
mlp/network_bnu_base.h
mlp/network_bnu_ref.h

convnet/layer.h
convnet/conv_layer.h
convnet/input_layer.h
convnet/output_layer.h
convnet/dropout_layer.h
convnet/pool_layer.h
convnet/tensor.h
convnet/tensor_solver.h
convnet/tensor_activations.h
convnet/tensor_loss_functions.h
convnet/tensor_operations.h
convnet/network.h
convnet/network_parallel.h
convnet/network_manager_convnet.h
convnet/network_interface_convnet.h
convnet/network_file_handler.h
convnet/tensor_gradient_checker.h
)

if (NOT APPLE)
	add_definitions( -Dcimg_use_png )
endif ()

add_library(neurocl SHARED ${sources_list} ${headers_list} ${sources_list_simd} ${headers_list_simd})

target_link_libraries(neurocl
boost_thread${boost_suffix}
boost_random${boost_suffix}
boost_system${boost_suffix}
boost_chrono${boost_suffix}
boost_filesystem${boost_suffix}
boost_serialization${boost_suffix}
${OpenCL_LIBRARIES}
X11
png
)
