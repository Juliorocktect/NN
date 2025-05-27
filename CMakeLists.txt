
cmake_minimum_required(VERSION 3.10)
project(VNNProjekt)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

# Library aus src und include
add_library(vnn_lib
    src/VNN.cpp
    src/ImageLoading.cpp
)

target_include_directories(vnn_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Executable
add_executable(main_exec main.cpp)


# Link die Bibliothek mit dem Executable
target_link_libraries(main_exec PRIVATE vnn_lib)
find_package(Eigen3 REQUIRED)
find_package(SDL3 REQUIRED)

target_link_libraries(main_exec PRIVATE Eigen3::Eigen)
target_link_libraries(main_exec PRIVATE SDL3::SDL3)
target_link_libraries(vnn_lib PRIVATE SDL3::SDL3)
target_link_libraries(vnn_lib PRIVATE Eigen3::Eigen)
#link eigen to targete
