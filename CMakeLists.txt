
cmake_minimum_required(VERSION 3.10)
project(VNNProjekt)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#find Eigen 
#list(APPEND CMAKE_INCLUDE_PATH "/usr/local/include")
#list(APPEND CMAKE_INCLUDE_PATH "/usr/local/share/eigen3/cmake/")
#LIST(APPEND CMAKE_MODULE_PATH "/usr/local/share/eigen3/cmake/Eigen3Config.cmake")
#find_package (Eigen3)

# Include dir for headers
include_directories(include)

# Library aus src und include
add_library(vnn_lib
    src/VNN.cpp
)

target_include_directories(vnn_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Executable
add_executable(main_exec main.cpp)


# Link die Bibliothek mit dem Executable
target_link_libraries(main_exec PRIVATE vnn_lib)
find_package(Eigen3 REQUIRED)
target_link_libraries(main_exec PRIVATE Eigen3::Eigen)
#link eigen to targete
