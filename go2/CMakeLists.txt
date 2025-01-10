cmake_minimum_required(VERSION 3.5)
project(go2deploy)
set(CMAKE_PREFIX_PATH "/opt/unitree_robotics;${CMAKE_PREFIX_PATH}")

message(STATUS "C Compiler: ${CMAKE_C_COMPILER}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")

find_package(pybind11 REQUIRED)
find_package(unitree_sdk2 REQUIRED)

## Set compiler to use c++ 17 features
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-lto")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-lto")
# set(CMAKE_INTERPROCEDURAL_OPTIMIZATION FALSE)

pybind11_add_module(go2py src/py.cpp)
target_link_libraries(go2py PRIVATE unitree_sdk2)
set_target_properties(go2py PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/go2deploy)