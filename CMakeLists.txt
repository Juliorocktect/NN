
cmake_minimum_required(VERSION 3.10)
project(VNNProjekt LANGUAGES CXX CUDA)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CPP_COMPILER clang)
#

include_directories(include)

# Library aus src und include
add_library(MATHS src/Matrix.cu src/GPUMatrix.cpp)
add_library(vnn_lib
    src/VNN.cpp
    src/ImageLoading.cpp
    src/backprop.cpp
    src/forwardProp.cpp
)

target_include_directories(vnn_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Executable
add_executable(main_exec main.cpp)

#get CUDA
set_target_properties(main_exec PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

find_package(CUDAToolkit REQUIRED)

#Link CUDA to project
target_link_libraries(main_exec PRIVATE CUDA::cudart)
target_link_libraries(MATHS PRIVATE CUDA::cudart)
# Link die Bibliothek mit dem Executable
target_link_libraries(main_exec PRIVATE MATHS)
target_link_libraries(main_exec PRIVATE vnn_lib)
target_link_libraries(vnn_lib PRIVATE MATHS)
find_package(Eigen3 REQUIRED)
find_package(SDL3 REQUIRED)

target_link_libraries(main_exec PRIVATE Eigen3::Eigen)
target_link_libraries(main_exec PRIVATE SDL3::SDL3)
target_link_libraries(MATHS PRIVATE Eigen3::Eigen)
target_link_libraries(vnn_lib PRIVATE SDL3::SDL3)
target_link_libraries(vnn_lib PRIVATE Eigen3::Eigen)
#link eigen to targete
