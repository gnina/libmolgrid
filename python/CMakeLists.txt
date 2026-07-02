
find_package(PythonLibs)
find_package(PythonInterp)
if (NOT PYTHONLIBS_FOUND)
  message(STATUS "Python libraries NOT found")
endif()
 
if(BUILD_CONTAINED_PYTHON)
  # create a python library with as few runtime dependencies as possible
  set(Boost_USE_STATIC_LIBS ON)
endif()

if(${CMAKE_VERSION} VERSION_LESS "3.12.0") 
	find_package( Boost COMPONENTS filesystem python${PYTHON_VERSION_MAJOR} REQUIRED )
else()
#minor version numbers requires newer (3.12?) version of cmake   
  find_package( Boost COMPONENTS filesystem python${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR} REQUIRED )
endif()

find_package( NumPy )

execute_process(
  COMMAND ${PYTHON_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.so', end='')"
  OUTPUT_VARIABLE PYTHON_MODULE_EXTENSION
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT PYTHON_MODULE_EXTENSION)
  set(PYTHON_MODULE_EXTENSION ".so")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${PYTHON_NUMPY_INCLUDE_DIR})

if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
#numpy header has unused function
    set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} -Wno-unused-function")
endif()

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/bindings.cpp COMMAND ${CMAKE_COMMAND} -DOUTNAME=${CMAKE_CURRENT_BINARY_DIR}/bindings.cpp -P set_docstrings.cmake
  DEPENDS bindings.in.cpp ${LIBMOLGRID_HEADERS} COMMENT "Inserting docstrings binding.cpp"
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

set( PYMOLGRID_SOURCES
 bindings.h
 ${CMAKE_CURRENT_BINARY_DIR}/bindings.cpp
 bindings_grids.cpp
)

set( PYMOLGRID_PY
 __init__.py
 torch_bindings.py
)



file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/molgrid/")

# link
# On macOS, Python extension modules must not link libpython directly:
# some distributions (e.g. conda's python) statically embed the
# interpreter in the executable rather than exposing libpython as a
# shared library, so linking against libpythonX.Y.dylib pulls in a
# second, independent copy of the CPython runtime and crashes at
# import time. Resolve Python symbols against the running process
# instead, as distutils/setuptools do.
if(APPLE)
  set(PYMOLGRID_PYTHON_LINK_LIBS)
  add_link_options("LINKER:-undefined,dynamic_lookup")
else()
  set(PYMOLGRID_PYTHON_LINK_LIBS ${PYTHON_LIBRARY})
endif()

if(BUILD_CONTAINED_PYTHON)
  # create a python library with as few runtime dependencies as possible
  add_link_options("LINKER:--exclude-libs,ALL")  # avoid exporting stdc++ symbols
  add_library(molgrid SHARED ${PYMOLGRID_SOURCES})
  target_link_libraries(molgrid ${Boost_LIBRARIES} ${PYMOLGRID_PYTHON_LINK_LIBS}
      ${CUDA_LIBRARIES} ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK} libmolgrid_static
      -static-libstdc++ -static-libgcc)
else()
# create the lib
  add_library(molgrid SHARED ${PYMOLGRID_SOURCES})
  target_link_libraries(molgrid ${Boost_LIBRARIES} ${PYMOLGRID_PYTHON_LINK_LIBS}
      ${CUDA_LIBRARIES} ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK} libmolgrid_shared)
endif()

# Suppress prefix "lib" because Python does not allow this prefix
set_target_properties(molgrid PROPERTIES 
									PREFIX ""
									SUFFIX "${PYTHON_MODULE_EXTENSION}"
									LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/molgrid/")
									
                  
# copy module code
foreach(file ${PYMOLGRID_PY})
 configure_file(${file}  ${CMAKE_CURRENT_BINARY_DIR}/molgrid/)
endforeach()

# Copy the __init__.py file
configure_file(setup.py ${CMAKE_CURRENT_BINARY_DIR}/setup.py)

#TODO: figure out reasonable install, also how to build both python2 and python3
install(CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/setup.py install --prefix ${CMAKE_INSTALL_PREFIX})")
