
find_package(PythonLibs)
find_package(PythonInterp)
if (NOT PYTHONLIBS_FOUND)
  message(STATUS "Python libraries NOT found")
endif()
 
if(${CMAKE_VERSION} VERSION_LESS "3.13.0") 
	find_package( Boost COMPONENTS system filesystem python${PYTHON_VERSION_MAJOR} REQUIRED )
else()
#minor version numbers requires newer (3.13?) version of cmake   
  find_package( Boost COMPONENTS system filesystem python${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR} REQUIRED )
endif()
#find_package( Boost COMPONENTS system filesystem python REQUIRED )

find_package( NumPy )

include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${PYTHON_NUMPY_INCLUDE_DIR})

if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
#numpy header has unused function
    set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} -Wno-unused-function")
endif()

add_custom_command(OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/bindings.cpp COMMAND ${CMAKE_COMMAND} -P set_docstrings.cmake 
  DEPENDS bindings.in.cpp COMMENT "Inserting docstrings binding.cpp"
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

set( PYMOLGRID_SOURCES
 bindings.h
 bindings.cpp
 bindings_grids.cpp
)

set( PYMOLGRID_PY
 __init__.py
 torch_bindings.py
)

# create the lib
add_library(molgrid SHARED ${PYMOLGRID_SOURCES})

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/molgrid/")
# Suppress prefix "lib" because Python does not allow this prefix
set_target_properties(molgrid PROPERTIES 
									PREFIX ""
									LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/molgrid/")
									
# link
target_link_libraries(molgrid ${Boost_LIBRARIES} ${CUDA_LIBRARIES} libmolgrid_shared)

# copy module code
foreach(file ${PYMOLGRID_PY})
 configure_file(${file}  ${CMAKE_CURRENT_BINARY_DIR}/molgrid/)
endforeach()

# Copy the __init__.py file
configure_file(setup.py ${CMAKE_CURRENT_BINARY_DIR}/setup.py)

#TODO: figure out reasonable install, also how to build both python2 and python3
install(CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/setup.py install --prefix ${CMAKE_INSTALL_PREFIX})")
