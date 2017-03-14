cmake_minimum_required (VERSION 2.8)
set(CMAKE_PREFIX_PATH "/usr/local/opencv3/share/OpenCV")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORCE_INLINES")
project(feature)
add_compile_options(-std=c++11)
set(OpenCV_DIR "/usr/local/opencv3")
find_package(OpenCV REQUIRED)

include_directories("/usr/local/opencv3/include")

add_executable(feature_extration feature_extration.cpp)
target_link_libraries(feature_extration ${OpenCV_LIBS})

add_executable(pose_estimation_2d2d pose_estimation_2d2d.cpp)
target_link_libraries(pose_estimation_2d2d ${OpenCV_LIBS} )

add_executable(pose_estimation_3d2d pose_estimation_3d2d)
target_link_libraries(pose_estimation_3d2d ${OpenCV_LIBS})
