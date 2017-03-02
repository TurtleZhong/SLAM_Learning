cmake_minimum_required (VERSION 2.8)
project (use_g2o)
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )
#modules
list (APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV REQUIRED)
include_directories(${Opencv_DIRS})
include_directories("/usr/local/include/eigen3")
# 寻找G2O
find_package( G2O REQUIRED )
include_directories( 
    ${G2O_INCLUDE_DIRS}
    "/usr/local/include/eigen3"
)

add_executable(useg2o useg2o.cpp)
target_link_libraries(useg2o ${OpenCV_LIBS} g2o_core g2o_stuff)
