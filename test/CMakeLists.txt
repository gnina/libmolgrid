

#Prep ourselves for compiling boost
find_package(Boost COMPONENTS unit_test_framework REQUIRED)
include_directories (${Boost_INCLUDE_DIRS})

#get all cpp files
set( TEST_SRCS
 test_grid.cpp
 test_grid.cu
 test_mgrid.cpp
 test_mgrid.cu
 test_quaternion.cpp
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

