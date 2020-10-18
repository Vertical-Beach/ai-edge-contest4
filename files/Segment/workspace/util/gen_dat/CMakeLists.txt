cmake_minimum_required(VERSION 3.0)
project(gen_dat)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

#--- enable output compile_command.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#--- $ cmake -DCMAKE_BUILD_TYPE=debug
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -MMD -Wall -Wextra -Winit-self")

#--- $ cmake
set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -Wall -O2 -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2 -march=native")

#--- Check C++14 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX14)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
  add_definitions(-DCOMPILEDWITHC14)
  message(STATUS "Using flag -std=c++14.")
elseif(COMPILER_SUPPORTS_CXX0X)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
  add_definitions(-DCOMPILEDWITHC0X)
  message(STATUS "Using flag -std=c++0x.")
else()
  message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++14 support. Please use a different C++ compiler.")
endif()

# can not use find-package in docker image
# because the cmake setting for OpenCV are not prepared
set(OpenCV_INCLUDE_DIRS
  /usr/local/include/opencv2
  /usr/local/include/opencv
  )
set(OpenCV_LIBS
  opencv_core
  opencv_imgcodecs
  opencv_highgui
  opencv_shape
  opencv_stitching
  opencv_objdetect
  opencv_superres
  opencv_videostab
  opencv_calib3d
  opencv_features2d
  opencv_highgui
  opencv_videoio
  opencv_imgcodecs
  opencv_video
  opencv_photo
  opencv_ml
  opencv_imgproc
  opencv_flann
  opencv_core
  )

#--- Build
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME}
  ./src/gen_dat.cpp
  )

target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBS}
  )
