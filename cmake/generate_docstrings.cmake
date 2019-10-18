# function to convert Doxygen-style comments into Python-style docstrings
function(doxyToDoc DOCSTRING DOXYSTRING)
  string(REGEX REPLACE "\brief " "" DOXSTRING"${DOXYSTRING}")
  if("${DOXYSTRING}" MATCHES "@param ([a-zA-Z0-9_]+) (.*)")
    set(DOXYSTRING ":param ${CMAKE_MATCH_1}: ${CMAKE_MATCH_2}")
  endif()
  if ("${DOXYSTRING}" MATCHES "@return (.+)")
    set(DOXYSTRING ":returns: ${CMAKE_MATCH_1}")
  endif()
  set(DOCSTRING ${DOCSTRING} ${DOXYSTRING} PARENT_SCOPE)
endfunction()

# automatically generate docstrings in bindings from Doxygen annotations in cpp header files
file(READ "bindings.cpp.in" CONTENTS)
STRING(REGEX REPLACE ";" "\\\\;" CONTENTS "${CONTENTS}")
string(REGEX REPLACE "\n" ";" CONTENTS "${CONTENTS}")

# make map of object name to line in bindings file 
# and list of object names for retrieval later
set(INDEX "0")
foreach(LINE ${CONTENTS})
  if("${LINE}" MATCHES "\"@(Docstring_[a-zA-Z0-9_]+)@\"\);")
    set(${CMAKE_MATCH_1} ${INDEX})
    set(OBJS ${OBJECTS} ${CMAKE_MATCH_1})
  endif()
  math(EXPR INDEX "${INDEX}+1")
endforeach()

# find the headers
file(GLOB HEADERS  ${PROJECT_SOURCE_DIR}/include/libmolgrid)

# make map of object name to docstring content
set(COPYING "0")
foreach(fname ${HEADERS})
  file(READ fname HEADER_CONTENTS)
  STRING(REGEX REPLACE ";" "\\\\;" HEADER_CONTENTS "${HEADER_CONTENTS}")
  string(REGEX REPLACE "\n" ";" HEADER_CONTENTS "${HEADER_CONTENTS}")
  foreach(LINE ${HEADER_CONTENTS})
    # extract Doxygen 
    if("${LINE}" MATCHES "// (Docstring_[a-zA-Z0-9_]+)")
      string(REGEX MATCH "@([a-z]+) (.*)" LINE "${LINE}")
      set(COPYING "1")
    # continuing extraction of previous comment
    elseif(${COPYING} AND "${LINE}" MATCHES "(/\*|\*)([\*]+)([a-zA-Z0-9@\\\.\(\)]+)")
      set(DOCSTRING ${DOCSTRING} ${LINE})
    # insert into map, zero out string
    elseif(${COPYING})
      string(REPLACE ";" "\\n" DOCSTRING "${DOCSTRING}")
      set(${CMAKE_MATCH_1}_DOXY ${DOCSTRING})
      set(COPYING "0")
      set(DOCSTRING "")
    endif()
  endforeach()
endforeach()

# To convert doxygen comments into docstrings
foreach(LINE ${DOXY_COMMENTS})
  string(REGEX REPLACE "\brief " "" LINE "${LINE}")
  if("${LINE}" MATCHES "@param ([a-zA-Z0-9_]+) (.*)")
    set(LINE ":param ${CMAKE_MATCH_1}: ${CMAKE_MATCH_2}")
  endif()
  if ("${LINE}" MATCHES "@return (.+)")
    set(LINE ":returns: ${CMAKE_MATCH_1}")
  endif()
  set(DOCSTRING ${DOCSTRING} ${LINE})
endforeach()
string(REPLACE ";" "\\n" DOCSTRING "${DOCSTRING}")

# insert docstrings into bindings
set(Docstring_${FUNCTION} ${DOCSTRING})
configure_file("bindings.cpp.in" "bindings.cpp" @ONLY)
