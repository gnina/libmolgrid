# creates the executable
add_executable(test_runner test_runner.cpp)
# indicates the include paths
target_include_directories(test_runner PRIVATE ${Boost_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
# indicates the shared library variant
target_compile_definitions(test_runner PRIVATE "BOOST_TEST_DYN_LINK=1")
# indicates the link paths
target_link_libraries(test_runner ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} ${Boost_PROGRAM_OPTIONS_LIBRARY} ${CUDA_LIBRARIES})

# declares a test with our executable
add_test(NAME testcpp COMMAND test_runner)
