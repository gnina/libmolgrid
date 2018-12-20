
set( LIBMOLGRID_SOURCES
 libmolgrid.cpp
)

set( LIBMOLGRID_HEADERS
 libmolgrid.h
)

add_library(libmolgrid_shared SHARED ${LIBMOLGRID_HEADERS} ${LIBMOLGRID_SOURCES})
SET_TARGET_PROPERTIES(libmolgrid_shared PROPERTIES OUTPUT_NAME molgrid CUDA_SEPARABLE_COMPILATION ON)

add_library(libmolgrid_static STATIC ${LIBMOLGRID_HEADERS} ${LIBMOLGRID_SOURCES})
SET_TARGET_PROPERTIES(libmolgrid_static PROPERTIES OUTPUT_NAME molgrid CUDA_SEPARABLE_COMPILATION ON)


#install libs
install(TARGETS libmolgrid_shared DESTINATION lib)
install(TARGETS libmolgrid_static DESTINATION lib)

#install headers
install(FILES ${LIBMOLGRID_HEADERS} DESTINATION include/libmolgrid)
