cmake_minimum_required(VERSION 3.25)
enable_testing()
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ compiler ID: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "C++ compiler version: ${CMAKE_CXX_COMPILER_VERSION}")

message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA host compiler: ${CMAKE_CUDA_HOST_COMPILER}")
# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#windows compile options
# set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin/nvcc.exe")
project(VNNProjekt LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)
include_directories(include)

#get test files
file(GLOB TEST_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp"
)


#get CUDA files
file(GLOB CUDA_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/*.cpp"
)

#models
file(GLOB MODEL_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/models/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/models/*.cpp"
)

#message(STATUS "CUDA_SOURCES = ${CUDA_SOURCES}")
# Library aus src und include
add_library(MATHS ${CUDA_SOURCES} ${MODEL_SOURCES})
add_library(vnn_lib
    src/algorithms/backPropNNG.cpp
    src/algorithms/forwardPropNNG.cpp
    src/algorithms/NNG.cpp
    src/utils/ImageLoading.cpp
)

target_include_directories(MATHS PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_include_directories(vnn_lib PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_include_directories(vnn_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Executable
add_executable(main_exec main.cpp)

#Set cuda arch for rtx 3060
set_target_properties(MATHS PROPERTIES
    CUDA_ARCHITECTURES 86
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

#get CUDA
set_target_properties(main_exec PROPERTIES
CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

#build test envoirenment
find_package(GTest REQUIRED)
add_executable(nn_tests ${TEST_FILES})
target_compile_features(nn_tests PRIVATE cxx_std_17)
set_target_properties(nn_tests PROPERTIES
    LINKER_LANGUAGE CXX
)
#set lang for libs
#target_compile_features(test PRIVATE cxx_std_17)
#Link CUDA to project

target_link_libraries(MATHS PUBLIC CUDA::cudart)
#target_link_libraries(test PRIVATE CUDA::cudart)
# Link die Bibliothek mit dem Executable
target_link_libraries(main_exec PRIVATE MATHS)
target_link_libraries(vnn_lib PRIVATE MATHS)
target_link_libraries(main_exec PRIVATE vnn_lib)
find_package(SDL3 REQUIRED)
include(GoogleTest)
gtest_discover_tests(nn_tests)

target_link_libraries(main_exec PRIVATE SDL3::SDL3)
target_link_libraries(vnn_lib PRIVATE SDL3::SDL3)
target_link_libraries(nn_tests PRIVATE
    GTest::gtest
    GTest::gtest_main
    MATHS
    /usr/lib/gcc/x86_64-linux-gnu/15/libstdc++.a
)
