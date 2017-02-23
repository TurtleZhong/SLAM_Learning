cmake_minimum_required (VERSION 2.8)
project(pcl_test)
set (CMAKE_CXX_FLAGS "-std=c++11")

find_package (OpenCV REQUIRED)
find_package (PCL REQUIRED)

include_directories ( ${OpenCV_INCLUDE_DIRS})
include_directories ( ${PCL_INCLUDE_DIRS} )
include_directories ("/usr/local/include/eigen3")
add_definitions(${PCL_DEFINITIONS})

add_executable(pcl_show pcl_show.cpp)
target_link_libraries(pcl_show ${PCL_LIBRARIES})

add_executable(pointCloud pointCloud.cpp)
target_link_libraries(pointCloud ${PCL_LIBRARIES} ${OpenCV_LIBS} )
