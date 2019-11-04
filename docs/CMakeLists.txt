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
    # process Doxyfile.in and substitute paths to generate a final Doxyfile
    CONFIGURE_FILE(${PROJECT_SOURCE_DIR}/docs/Doxyfile.in
      ${PROJECT_SOURCE_DIR}/Doxyfile
    )
    # remove any stale documentation that might already exist in the docs directory
    file(GLOB_RECURSE DOCFILES ${PROJECT_SOURCE_DIR}/docs/_docs/*)
    foreach(file ${DOCFILES})
      file(REMOVE "${file}")
    endforeach()
    # generate jekyll-compatible html from notebooks for the tutorials page
    file(GLOB NOTEBOOKS ${PROJECT_SOURCE_DIR}/docs/_tutorials/*.ipynb)
    foreach(file ${NOTEBOOKS})
      get_filename_component(BASE ${file} NAME_WE)
      set(OUTNAME ${PROJECT_SOURCE_DIR}/docs/_tutorials/${BASE}.html)
      string(REPLACE "_" " " TITLE ${BASE})
      execute_process(COMMAND bash "-c" "echo ${TITLE} | sed 's/.*/\\L&/; s/[a-z]*/\\u&/'" OUTPUT_VARIABLE TITLE)
      execute_process(COMMAND bash "-c" "echo '---\ntitle: ${TITLE}\nlayout: tutorials\nhomepage: false\nhide: true\n---\n' > ${OUTNAME}")
      execute_process(COMMAND bash "-c" "jupyter nbconvert --stdout ${file} >> ${OUTNAME}")
    endforeach()
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
