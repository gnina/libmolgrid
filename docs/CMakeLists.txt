OPTION(BUILD_DOCS "Build libmolgrid documentation" OFF)
IF(BUILD_DOCS)
    find_package(Doxygen)
    if(NOT DOXYGEN_EXECUTABLE)
			message(WARNING "Doxygen not found; docs will not be built")
      return()
    endif(NOT DOXYGEN_EXECUTABLE)
    IF(DOT)
        SET(HAVE_DOT YES)
    ELSE(DOT)
        SET(HAVE_DOT NO)
    ENDIF(DOT)
    find_package(Sphinx)
		if(NOT SPHINX_EXECUTABLE)
			message(WARNING "Sphinx not found; docs will not be built")
    	return()
		endif(NOT SPHINX_EXECUTABLE)

		# Find all the public headers
		file(GLOB_RECURSE LIBMOLGRID_PUBLIC_HEADERS ${PROJECT_SOURCE_DIR}/include/libmolgrid/*.h)

		set(DOXYGEN_INPUT_DIR ${PROJECT_SOURCE_DIR})
		set(DOXYGEN_OUTPUT_DIR ${PROJECT_SOURCE_DIR}/docs/cpp/xml)
    file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR})
    set(DOXYGEN_INDEX_FILE ${DOXYGEN_OUTPUT_DIR}/index.xml)
		set(DOXYFILE_IN ${PROJECT_SOURCE_DIR}/docs/Doxyfile.in)
		set(DOXYFILE_OUT ${PROJECT_SOURCE_DIR}/Doxyfile)
		 
    # process Doxyfile.in and substitute paths to generate a final Doxyfile
		configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT})
		 
		# only regenerate Doxygen when the Doxyfile or public headers change
		add_custom_command(OUTPUT ${DOXYGEN_INDEX_FILE}
		                   DEPENDS ${LIBMOLGRID_PUBLIC_HEADERS}
		                   COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
		                   MAIN_DEPENDENCY ${DOXYFILE_OUT} ${DOXYFILE_IN}
		                   COMMENT "Generating docs"
		                   VERBATIM)
		 
		add_custom_target(Doxygen ALL DEPENDS ${DOXYGEN_INDEX_FILE})
		 
		set(SPHINX_SOURCE ${PROJECT_SOURCE_DIR}/sphinx/cpp)
		set(SPHINX_BUILD ${PROJECT_SOURCE_DIR}/docs/cpp)
		set(SPHINX_INDEX_FILE ${SPHINX_BUILD}/index.html)
		 
		# Only regenerate Sphinx when:
		# - Doxygen has rerun
		# - Our doc files have been updated
		# - The Sphinx config has been updated
		add_custom_command(OUTPUT ${SPHINX_INDEX_FILE}
		                   COMMAND 
		                     ${SPHINX_EXECUTABLE} -b html
		                     # Tell Breathe where to find the Doxygen output
		                     -Dbreathe_projects.ligmolgrid=${DOXYGEN_OUTPUT_DIR}
		                   ${SPHINX_SOURCE} ${SPHINX_BUILD}
		                   WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/docs/cpp
		                   DEPENDS
		                   ${PROJECT_SOURCE_DIR}/sphinx/cpp/index.rst
		                   ${DOXYGEN_INDEX_FILE}
		                   MAIN_DEPENDENCY ${SPHINX_SOURCE}/conf.py
                       COMMENT "Generating C++ documentation with Sphinx")
		 
		add_custom_target(Sphinx ALL DEPENDS ${SPHINX_INDEX_FILE})

    # update sphinx output to include bootstrap layout
    set(SPHINX_HTML_OUTPUT
      "${SPHINX_BUILD}/genindex.html;${SPHINX_BUILD}/index.html;${SPHINX_BUILD}/search.html")
    foreach(fname ${SPHINX_HTML_OUTPUT})
      add_custom_command(TARGET Sphinx POST_BUILD COMMAND bash "-c" "sed -i '1s/^/---\\\\ntitle: libmolgrid C++ Documentation\\\\nlayout: default\\\\nhomepage: false\\\\nhide: true\\\\n---\\\\n/' ${fname}")
    endforeach()
    add_custom_command(TARGET Sphinx POST_BUILD COMMAND bash "-c" "sed -i 's/wy-nav-side{position:fixed\;top\:0/wy-nav-side{position\:fixed\;top\:34\.25%/' ${SPHINX_BUILD}/_static/css/theme.css")
		 
    # generate jekyll-compatible html from notebooks for the tutorials page
    file(GLOB NOTEBOOKS ${PROJECT_SOURCE_DIR}/docs/_tutorials/*.ipynb)
    foreach(file ${NOTEBOOKS})
      get_filename_component(BASE ${file} NAME_WE)
      set(OUTNAME ${PROJECT_SOURCE_DIR}/docs/_tutorials/${BASE}.html)
      set(TMPFILE ${PROJECT_SOURCE_DIR}/docs/_tutorials/tmp)
      string(REPLACE "_" " " TITLE ${BASE})
      execute_process(COMMAND bash "-c" "echo ${TITLE} | sed 's/.*/\\L&/; s/[a-z]*/\\u&/'" OUTPUT_VARIABLE TITLE)
      execute_process(COMMAND bash "-c" "echo '---\ntitle: ${TITLE}layout: tutorials\nhomepage: false\nhide: true\n---\n' > ${OUTNAME}")
      execute_process(COMMAND bash "-c" "sed 's/\\\\n/jocelynsnewline\\\\n/g' ${file} > ${TMPFILE}")
      execute_process(COMMAND bash "-c" "jupyter nbconvert --stdout ${TMPFILE} >> ${OUTNAME}")
      execute_process(COMMAND bash "-c" "sed -i 's/jocelynsnewline/<br>/g' ${OUTNAME}")
      execute_process(COMMAND bash "-c" "sed -i 's/<div class=\"container\" id=\"notebook-container\">/<div class=\"bs-docs-container row\">\\n<div class=col-md-9\" role=\"main\">/g' ${OUTNAME}")
      execute_process(COMMAND bash "-c" "sed -i 's/<\\/body>/\\n<\\/div>\\n<\\/body>/' ${OUTNAME}")
      execute_process(COMMAND bash "-c" "sed -i 's/\\(.*img src.*\\)<br>/\\1/g' ${OUTNAME}")
      file(REMOVE ${TMPFILE})
    endforeach()
ENDIF(BUILD_DOCS)
