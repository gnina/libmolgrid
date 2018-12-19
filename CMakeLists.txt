cmake_minimum_required (VERSION 3.5)
project(LibMolGrid C CXX )

#version number
set (VERSION_MAJOR 0)
set (VERSION_MINOR 1)
set (VERSION_PATCH 0)

# get git hash
include(cmake/git_revision.cmake)

# guard against in-source builds and bad build-type strings
include(cmake/safeguards.cmake)

# place binaries and libraries according to GNU standards
include(GNUInstallDirs)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

#dependencies
find_package(CUDA REQUIRED)
find_package(Boost REQUIRED COMPONENTS python regex unit_test_framework program_options)
find_package(OpenBabel2 REQUIRED)

# configure a header file to pass some of the CMake settings
# to the source code
configure_file (
  "${PROJECT_SOURCE_DIR}/cmake/config.h.in"
  "${PROJECT_BINARY_DIR}/config.h"
  )



# add the binary tree to the search path for include files
# so that we will find libmolgrid_config.h
include_directories("${PROJECT_BINARY_DIR}")
include_directories(${CMAKE_SOURCE_DIR}/src)

add_subdirectory(src)
add_subdirectory(python)

# enable testing
include(CTest)
enable_testing()

# define tests
add_subdirectory(test)
