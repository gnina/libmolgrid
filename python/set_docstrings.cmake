# generate Python docstrings from Cpp source; call with -DOUTNAME=<output file name>
set(PROJECT_SOURCE_DIR "..")
# function to convert Doxygen-style comments into Python-style docstrings
function(doxyToDoc DOXYSTRING)
  string(REGEX REPLACE "\\\\brief[ \t]*" "" DOXYSTRING "${DOXYSTRING}")
  string(REGEX REPLACE "\\\\class[ \t]*" "" DOXYSTRING "${DOXYSTRING}")
  string(REGEX REPLACE "\"" "\\\\\"" DOXYSTRING "${DOXYSTRING}")

  if("${DOXYSTRING}" MATCHES "@param[ \t]*(\\[[a-zA-Z0-9_]+\\]) ([A-Za-z0-9_]+)(.*)")
    set(DOXYSTRING ":param ${CMAKE_MATCH_2}: ${CMAKE_MATCH_3}")
  elseif ("${DOXYSTRING}" MATCHES "@return (.+)")
    set(DOXYSTRING ":returns: ${CMAKE_MATCH_1}")
  elseif("${DOXYSTRING}" MATCHES "(.*)(\\*/)")
    set(DOXYSTRING "${CMAKE_MATCH_1}")
  endif()
  set(DOCSTRING "${DOCSTRING}" "${DOXYSTRING}" PARENT_SCOPE)
endfunction()

# find the headers
file(GLOB HEADERS  ${PROJECT_SOURCE_DIR}/include/libmolgrid/*.h)

# make map of object name to docstring content
set(COPYING "0")
foreach(fname ${HEADERS})
  file(READ ${fname} HEADER_CONTENTS)
  STRING(REGEX REPLACE ";" "\\\\;" HEADER_CONTENTS "${HEADER_CONTENTS}")
  string(REGEX REPLACE "\n" ";" HEADER_CONTENTS "${HEADER_CONTENTS}")
  foreach(LINE ${HEADER_CONTENTS})
    # extract Doxygen 
    if("${LINE}" MATCHES "[ \t]*// (Docstring_[a-zA-Z0-9_]+)")
      set(COPYING "1")
      set(FUNC ${CMAKE_MATCH_1})
    # continuing extraction of previous comment
    elseif(${COPYING})
      if("${LINE}" MATCHES "([ \t]*)(/\\*|\\*)(\\**)(.*)")
        doxyToDoc("${CMAKE_MATCH_4}")
      else()
        # insert into map, zero out string
        string(REPLACE ";" "\\n" DOCSTRING "${DOCSTRING}")
        set(${FUNC} ${DOCSTRING})
        set(COPYING "0")
        set(DOCSTRING "")
      endif()
    endif()
  endforeach()
endforeach()

set("THIS_IS_THE_SOURCE_FILE" "\n\n  WARNING WARNING WARNING\n  This file is generated from bindings.in.cpp\n\nDO NOT EDIT\n\n")
# insert docstrings into bindings

configure_file("${PROJECT_SOURCE_DIR}/python/bindings.in.cpp" "${OUTNAME}" @ONLY)
