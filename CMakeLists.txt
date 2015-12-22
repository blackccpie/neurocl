cmake_minimum_required (VERSION 2.8)
project (neurocl)

include_directories(. "../vexcl")

include("../vexcl/cmake/opencl/FindOpenCL.cmake")

set (sources_list
network.cpp
network_bnu.cpp
main.cpp
)

set (headers_list
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
