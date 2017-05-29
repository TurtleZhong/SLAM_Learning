CMAKE_MINIMUM_REQUIRED( VERSION 2.8 )
set(CMAKE_PREFIX_PATH "/usr/local/opencv3/share/OpenCV")
PROJECT( vo_direct )

set(OpenCV_DIR "/usr/local/opencv3")
find_package(OpenCV REQUIRED)

include_directories("/usr/local/opencv3/include")

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
# PCL
FIND_PACKAGE( PCL REQUIRED COMPONENTS common io visualization filters )
list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4") # use this in Ubuntu 16.04

# 添加头文件和库文件
ADD_DEFINITIONS( ${PCL_DEFINITIONS} )
INCLUDE_DIRECTORIES( ${PCL_INCLUDE_DIRS}  )
LINK_LIBRARIES( ${PCL_LIBRARY_DIRS} )

# OpenCV
#find_package( OpenCV 3.2 REQUIRED )
#include_directories( ${OpenCV_INCLUDE_DIRS} )
# Sophus 
find_package( Sophus REQUIRED )
include_directories( ${Sophus_INCLUDE_DIRS} )

# G2O
find_package( G2O REQUIRED )
include_directories( ${G2O_INCLUDE_DIRS} )

set( THIRD_PARTY_LIBS 
    ${OpenCV_LIBS}
    ${Sophus_LIBRARIES}
    ${PCL_LIBRARIES}
    g2o_core g2o_stuff g2o_types_sba g2o_solver_csparse g2o_csparse_extension
)

set( THIRD_PARTY_LIBS 
    ${OpenCV_LIBS}
    ${Sophus_LIBRARIES}
    ${PCL_LIBRARIES}
    g2o_core g2o_stuff g2o_types_sba g2o_solver_csparse g2o_csparse_extension
)


ADD_SUBDIRECTORY( ${PROJECT_SOURCE_DIR}/src )
ADD_SUBDIRECTORY( ${PROJECT_SOURCE_DIR}/test )
