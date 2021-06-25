
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
	find_package( Boost COMPONENTS system filesystem python${PYTHON_VERSION_MAJOR} REQUIRED )
else()
#minor version numbers requires newer (3.12?) version of cmake   
  find_package( Boost COMPONENTS system filesystem python${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR} REQUIRED )
endif()

find_package( NumPy )

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
if(BUILD_CONTAINED_PYTHON)
  # create a python library with as few runtime dependencies as possible
  add_link_options("LINKER:--exclude-libs,ALL")  # avoid exporting stdc++ symbols
  add_library(molgrid SHARED ${PYMOLGRID_SOURCES})
  target_link_libraries(molgrid ${Boost_LIBRARIES} ${CUDA_LIBRARIES} libmolgrid_static  -static-libstdc++ -static-libgcc)
else()
# create the lib
  add_library(molgrid SHARED ${PYMOLGRID_SOURCES})
  target_link_libraries(molgrid ${Boost_LIBRARIES} ${CUDA_LIBRARIES} libmolgrid_shared)
endif()

# Suppress prefix "lib" because Python does not allow this prefix
set_target_properties(molgrid PROPERTIES 
									PREFIX ""
									LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/molgrid/")
									
                  
# copy module code
foreach(file ${PYMOLGRID_PY})
 configure_file(${file}  ${CMAKE_CURRENT_BINARY_DIR}/molgrid/)
endforeach()

# Copy the __init__.py file
configure_file(setup.py ${CMAKE_CURRENT_BINARY_DIR}/setup.py)

#TODO: figure out reasonable install, also how to build both python2 and python3
install(CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/setup.py install --prefix ${CMAKE_INSTALL_PREFIX})")
