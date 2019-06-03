cmake_minimum_required(VERSION 2.6)
# set( CMAKE_VERBOSE_MAKEFILE on )

project(imageFilter)

# declare executable file and source files
add_executable(
        imageCopyFilter
        imageCopyFilter.cpp
	utils.cpp
)

# use this to suppress warnings on deprecated functions
set (CMAKE_CXX_FLAGS "-Wno-deprecated-declarations" )

target_link_libraries(
  imageCopyFilter
  OpenCL
  m
  freeimage
)
