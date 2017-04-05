cmake_minimum_required(VERSION 2.8)
project(feature_detect)
find_package(OpenCV 2.4.11 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(feature_detect src/feature_detect.cpp)
target_link_libraries(feature_detect ${OpenCV_LIBS})
