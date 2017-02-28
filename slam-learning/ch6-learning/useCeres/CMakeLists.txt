cmake_minimum_required (VERSION 2.8)
project(optimiztion)
set (CMAKE_CXX_FLAGS "-std=c++11")

find_package (OpenCV REQUIRED)
find_package (Ceres REQUIRED)

include_directories (${OpenCV_INCLUDE_DIES})
include_directories (${Ceres_INCLUDE_DIRS})

add_executable (useCeres useCeres.cpp)
target_link_libraries (useCeres ${OpenCV_LIBS} ${Ceres_LIBS})
