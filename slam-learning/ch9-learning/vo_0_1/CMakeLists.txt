CMAKE_MINIMUM_REQUIRED( VERSION 2.8 )
#set(CMAKE_PREFIX_PATH "/usr/local/opencv3/share/OpenCV")
PROJECT( vo_0_1 )

#set(OpenCV_DIR "/usr/local/opencv3")
#find_package(OpenCV REQUIRED)

#include_directories("/usr/local/opencv3/include")

SET(CMAKE_CXX_COMPILER "g++")
SET( CMAKE_BUILD_TYPE Release )
SET( CMAKE_CXX_FLAGS "-O3 -std=c++11" )
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

INCLUDE_DIRECTORIES( ${PROJECT_SOURCE_DIR}/include )
LINK_DIRECTORIES( ${PROJECT_SOURCE_DIR}/lib)

############### dependencies ######################
# Eigen
include_directories( "/usr/include/eigen3" )
# OpenCV
find_package( OpenCV 3.2 REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
# Sophus 
find_package( Sophus REQUIRED )
include_directories( ${Sophus_INCLUDE_DIRS} )

# G2O
find_package( G2O REQUIRED )
include_directories( ${G2O_INCLUDE_DIRS} )

set( THIRD_PARTY_LIBS 
    ${OpenCV_LIBS}
    ${Sophus_LIBRARIES}
    g2o_core g2o_stuff g2o_types_sba
)

set( THIRD_PARTY_LIBS 
    ${OpenCV_LIBS}
    ${Sophus_LIBRARIES}
    g2o_core g2o_stuff g2o_types_sba
)


ADD_SUBDIRECTORY( ${PROJECT_SOURCE_DIR}/src )
ADD_SUBDIRECTORY( ${PROJECT_SOURCE_DIR}/test )
