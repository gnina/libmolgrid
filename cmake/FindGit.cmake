find_program(
    GIT_EXECUTABLE
    NAMES git
    DOC "git command line client"
    )
mark_as_advanced(GIT_EXECUTABLE)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Git DEFAULT_MSG GIT_EXECUTABLE)
