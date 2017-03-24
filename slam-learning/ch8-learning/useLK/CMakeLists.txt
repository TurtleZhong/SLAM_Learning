cmake_minimum_required(VERSION 2.8)
project(useLK)
add_compile_options(-std=c++11)

find_package(OpenCV REQUIRED)

add_executable(use_lk use_lk.cpp)
target_link_libraries(use_lk ${OpenCV_LIBS})
