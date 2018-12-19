find_package(Git)

if(GIT_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-list --max-count=1 HEAD
        OUTPUT_VARIABLE GIT_REVISION
        ERROR_QUIET
        )
    if(NOT ${GIT_REVISION} STREQUAL "")
        string(STRIP ${GIT_REVISION} GIT_REVISION)
    endif()
    message(STATUS "Current git revision is ${GIT_REVISION}")
else()
    set(GIT_REVISION "unknown")
endif()
