OPTION(BUILD_DOCS "Build libmolgrid documentation" OFF)
IF(BUILD_DOCS)
    find_package(Doxygen)
    if(NOT DOXYGEN_EXECUTABLE)
      return()
    endif(NOT DOXYGEN_EXECUTABLE)
    IF(DOT)
        SET(HAVE_DOT YES)
    ELSE(DOT)
        SET(HAVE_DOT NO)
    ENDIF(DOT)
    # This processes our Doxyfile.in and substitutes paths to generate
    # a final Doxyfile
    CONFIGURE_FILE(${PROJECT_SOURCE_DIR}/docs/Doxyfile.in
      ${PROJECT_SOURCE_DIR}/Doxyfile
    )
    # This creates a new target to build documentation.
    # It runs ${DOXYGEN} which is the full path and executable to
    # Doxygen on your system, set by the FindDoxygen.cmake module
    # It runs the final generated Doxyfile against it.
    # The DOT_PATH is substituted into the Doxyfile.
    ADD_CUSTOM_TARGET(docs ALL 
      COMMAND ${DOXYGEN_EXECUTABLE} ${PROJECT_SOURCE_DIR}/Doxyfile
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    )
ENDIF(BUILD_DOCS)
