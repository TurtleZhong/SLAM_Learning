cmake_minimum_required (VERSION 2.8)

project(feature)
set(OpenCV_DIR "/usr/local/opencv3")
find_package(OpenCV REQUIRED)

include_directories("/usr/local/opencv3/include")

add_executable(feature_extration feature_extration.cpp)
target_link_libraries(feature_extration ${OpenCV_LIBS})

#add_executable(pose_estimation_2d2d pose_estimation_2d2d.cpp)
#target_link_libraries(pose_estimation_2d2d ${OpenCV_LIBS} )
