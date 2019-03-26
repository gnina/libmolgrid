

#Prep ourselves for compiling boost
find_package(Boost COMPONENTS unit_test_framework system REQUIRED)
include_directories (${Boost_INCLUDE_DIRS})

#get all cpp files
set( TEST_SRCS
 test_grid.cpp
 test_grid.cu
 test_mgrid.cpp
 test_mgrid.cu
 test_quaternion.cpp
 test_transform.cpp
 test_transform.cu
)

#Run through each source
foreach(testSrc ${TEST_SRCS})
        #Extract the filename without an extension (NAME_WE)
        get_filename_component(fName ${testSrc} NAME)
        string(REPLACE "." "_" testName ${fName})

        #Add compile target
        add_executable(${testName} ${testSrc})

		target_compile_definitions(${testName} PRIVATE "BOOST_TEST_DYN_LINK=1")

        #link to Boost libraries AND your targets and dependencies
        target_link_libraries(${testName} libmolgrid_static ${Boost_LIBRARIES} ${CUDA_LIBRARIES})

        #Finally add it to test execution - 

        add_test(NAME ${testName} COMMAND ${testName})
endforeach(testSrc)

# find python
find_package(PythonInterp)
if(NOT PYTHONINTERP_FOUND)
    message(FATAL_ERROR "ERROR: Python interpreter not found. Cannot continue - sorry.")
else()
	#check for pytest
	execute_process(COMMAND ${PYTHON_EXECUTABLE} -m pytest --version
	  OUTPUT_VARIABLE PYTEST_output
	  ERROR_VARIABLE  PYTEST_error
	  RESULT_VARIABLE PYTEST_result)
	if(${PYTEST_result})
	  message(SEND_ERROR "Pytest package not available: ${PYTEST_error}")
	endif()

	add_test(NAME pymolgrid COMMAND ${PYTHON_EXECUTABLE} -m pytest -rsv WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
	set_tests_properties(pymolgrid PROPERTIES ENVIRONMENT
			"PYTHONPATH=${CMAKE_BINARY_DIR}/python:$PYTHONPATH")
endif()