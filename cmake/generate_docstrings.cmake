# function to convert Doxygen-style comments into Python-style docstrings
function(doxyToDoc DOXYSTRING)
  string(REGEX REPLACE "[ \t]*/\\*+[ \t]*\\\\brief[ \t]*" "" DOXYSTRING "${DOXYSTRING}")
  if("${DOXYSTRING}" MATCHES "[ \t]*\\*[ \t]*@param[ \t]*(\\[[a-zA-Z0-9_]+\\]) (.*)")
    set(DOXYSTRING ":param ${CMAKE_MATCH_1}: ${CMAKE_MATCH_2}")
  elseif ("${DOXYSTRING}" MATCHES "[ \t]*\\*[ \t]*@return (.+)")
    set(DOXYSTRING ":returns: ${CMAKE_MATCH_1}")
  elseif("${DOXYSTRING}" MATCHES "[ \t]*\\*/*(.*)")
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
    if("${LINE}" MATCHES "[ \t]+// (Docstring_[a-zA-Z0-9_]+)")
      set(COPYING "1")
      set(FUNC ${CMAKE_MATCH_1})
    # continuing extraction of previous comment
    elseif(${COPYING})
      if("${LINE}" MATCHES "([ \t]*)(/\\*|\\*)(\\**)(.*)")
        doxyToDoc("${LINE}")
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

# insert docstrings into bindings
configure_file("${PROJECT_SOURCE_DIR}/python/bindings.cpp.in" "${PROJECT_SOURCE_DIR}/python/bindings.cpp" @ONLY)
