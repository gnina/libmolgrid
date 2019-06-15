
set( LIBMOLGRID_SOURCES
 libmolgrid.cpp
 atom_typer.cpp
 example.cpp
 exampleref_providers.cpp
 example_extractor.cpp
 example_provider.cpp
 grid_maker.cpp
 grid_maker.cu
 coordinateset.cpp
 coord_cache.cpp
 transform.cpp
 transform.cu
 grid_io.cpp
 cartesian_grid.cpp
)

set( LIBMOLGRID_HEADERS
 ../include/libmolgrid/libmolgrid.h
 ../include/libmolgrid/grid.h
 ../include/libmolgrid/managed_grid.h
 ../include/libmolgrid/quaternion.h
 ../include/libmolgrid/transform.h
 ../include/libmolgrid/atom_typer.h
 ../include/libmolgrid/example.h
 ../include/libmolgrid/coordinateset.h
 ../include/libmolgrid/exampleref_providers.h
 ../include/libmolgrid/example_extractor.h
 ../include/libmolgrid/example_provider.h
 ../include/libmolgrid/grid_maker.h
 ../include/libmolgrid/coord_cache.h
 ../include/libmolgrid/common.h
 ../include/libmolgrid/grid_io.h
 ../include/libmolgrid/cartesian_grid.h
)

#include_directories (${Boost_INCLUDE_DIRS})
#include_directories(${OPENBABEL2_INCLUDE_DIR})

add_library(libmolgrid_shared SHARED ${LIBMOLGRID_HEADERS} ${LIBMOLGRID_SOURCES})
SET_TARGET_PROPERTIES(libmolgrid_shared PROPERTIES OUTPUT_NAME molgrid CUDA_SEPARABLE_COMPILATION OFF)

add_library(libmolgrid_static STATIC ${LIBMOLGRID_HEADERS} ${LIBMOLGRID_SOURCES})
SET_TARGET_PROPERTIES(libmolgrid_static PROPERTIES OUTPUT_NAME molgrid CUDA_SEPARABLE_COMPILATION OFF)

target_link_libraries(libmolgrid_shared ${OPENBABEL2_LIBRARIES} ${Boost_LIBRARIES})
target_link_libraries(libmolgrid_static ${OPENBABEL2_LIBRARIES} ${Boost_LIBRARIES})

#install libs
install(TARGETS libmolgrid_shared DESTINATION lib)
install(TARGETS libmolgrid_static DESTINATION lib)

#install headers
install(FILES ${LIBMOLGRID_HEADERS} ${PROJECT_BINARY_DIR}/include/libmolgrid/config.h DESTINATION include/libmolgrid)
