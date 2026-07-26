cmake_minimum_required(VERSION 3.10)


# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#windows compile options
# set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin/nvcc.exe")
project(VNNProjekt LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)
include_directories(include)

#get CUDA files
file(GLOB CUDA_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/*.cpp"
)
#message(STATUS "CUDA_SOURCES = ${CUDA_SOURCES}")
# Library aus src und include
add_library(MATHS src/Matrix.cu ${CUDA_SOURCES} src/GPUMatrix.cpp )
add_library(vnn_lib
    src/backPropNNG.cpp
    src/forwardPropNNG.cpp
    src/NNG.cpp
    src/ImageLoading.cpp
)

target_include_directories(MATHS PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_include_directories(vnn_lib PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_include_directories(vnn_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Executable
add_executable(main_exec main.cpp)

#Set cuda arch for rtx 3060
set_target_properties(MATHS PROPERTIES
    CUDA_ARCHITECTURES 86
)

#get CUDA
#set_target_properties(main_exec PROPERTIES
#CUDA_SEPARABLE_COMPILATION ON
#)
set_target_properties(MATHS PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

find_package(GTest REQUIRED)
#add_executable(test tests/test_main.cpp)

#set lang for libs
#target_compile_features(test PRIVATE cxx_std_17)
#Link CUDA to project
target_link_libraries(main_exec PRIVATE CUDA::cudart)
target_link_libraries(MATHS PRIVATE CUDA::cudart)
#target_link_libraries(test PRIVATE CUDA::cudart)
# Link die Bibliothek mit dem Executable
target_link_libraries(main_exec PRIVATE MATHS)
target_link_libraries(vnn_lib PRIVATE MATHS)
target_link_libraries(main_exec PRIVATE vnn_lib)
find_package(SDL3 REQUIRED)

target_link_libraries(main_exec PRIVATE SDL3::SDL3)
target_link_libraries(vnn_lib PRIVATE SDL3::SDL3)
#target_link_libraries(test PRIVATE
#    GTest::gtest
#    GTest::gtest_main
#    MATHS
#    CUDA::cudart
#)
