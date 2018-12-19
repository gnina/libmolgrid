# find python
find_package(PythonInterp)
if(NOT PYTHONINTERP_FOUND)
    message(FATAL_ERROR "ERROR: Python interpreter not found. Cannot continue - sorry.")
endif()

find_package(PythonLibs)
if (NOT PYTHONLIBS_FOUND)
  message(STATUS "Python libraries NOT found")
endif()
    
