cmake_minimum_required (VERSION 3.18)
project(LibMolGrid LANGUAGES C CXX CUDA)

#version number
set (VERSION_MAJOR 0)
set (VERSION_MINOR 5)
set (VERSION_PATCH 3)

#set default build type
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_USE_RESPONSE_FILE_FOR_INCLUDES OFF)

option(BUILD_SHARED "Build shared library" ON)
option(BUILD_STATIC "Build static library" ON)
option(BUILD_COVERAGE "Build with code coverage" OFF)

if(${BUILD_CONTAINED_PYTHON}) 
  #if make a mostly statically linked python module, use static boost
   set(BUILD_STATIC ON)
   set(Boost_USE_STATIC_LIBS ON)
endif()

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/Modules;${PROJECT_SOURCE_DIR}/cmake")
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
find_package(Boost REQUIRED COMPONENTS regex unit_test_framework program_options system filesystem iostreams serialization)
find_package(OpenBabel3 REQUIRED)
include_directories(SYSTEM ${OPENBABEL3_INCLUDE_DIR})
find_package(ZLIB)

# configure a header file to pass some of the CMake settings
# to the source code
configure_file (
  "${PROJECT_SOURCE_DIR}/cmake/config.h.in"
  "${PROJECT_BINARY_DIR}/include/libmolgrid/config.h"
  )

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
    set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} -Wall -Wno-unknown-pragmas -Werror")
    set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g3")
    if (BUILD_COVERAGE)
	set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g -fprofile-arcs -ftest-coverage")
    else()
        set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g")
    endif()	
endif()

if(CUDA_VERSION_MAJOR LESS 11)
#compile for all major architectures >= 35
 set(CMAKE_CUDA_ARCHITECTURES 35 50 60 70 75)
else()
 set(CMAKE_CUDA_ARCHITECTURES 52 60 70 75 80)
endif()

set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -g -lineinfo")
set(CMAKE_CUDA_FLAGS_DEBUG "-O0 -g -G")

# add the binary tree to the search path for include files
# so that we will find libmolgrid_config.h
include_directories("${PROJECT_BINARY_DIR}")
include_directories(BEFORE ${CMAKE_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
include_directories(SYSTEM ${Boost_INCLUDE_DIR})

add_subdirectory(src)
add_subdirectory(python)
add_subdirectory(docs)

# enable testing
include(CTest)
enable_testing()
# define tests
add_subdirectory(test)
