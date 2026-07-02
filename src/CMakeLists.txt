
set( LIBMOLGRID_SOURCES
 libmolgrid.cpp
 atom_typer.cpp
 example.cpp
 exampleref_providers.cpp
 example_extractor.cpp
 example_provider.cpp
 example_dataset.cpp
 grid_maker.cpp
 coordinateset.cpp
 coord_cache.cpp
 transform.cpp
 grid_io.cpp
 grid_interpolater.cpp
 cartesian_grid.cpp
 string_cache.cpp
)

if(LIBMOLGRID_USE_CUDA)
 list(APPEND LIBMOLGRID_SOURCES
  grid_maker.cu
  coordinateset.cu
  transform.cu
  grid_interpolater.cu
 )
else()
 list(APPEND LIBMOLGRID_SOURCES
  grid_maker_gpu.mm
  coordinateset_gpu.mm
  transform_gpu.mm
  grid_interpolater_gpu.mm
  metal_context.mm
 )
endif()

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
 ../include/libmolgrid/example_dataset.h
 ../include/libmolgrid/grid_maker.h
  ../include/libmolgrid/grid_interpolater.h
 ../include/libmolgrid/coord_cache.h
 ../include/libmolgrid/common.h
 ../include/libmolgrid/grid_io.h
 ../include/libmolgrid/cartesian_grid.h
 ../include/libmolgrid/string_cache.h
)

if(LIBMOLGRID_USE_METAL)
 list(APPEND LIBMOLGRID_HEADERS ../include/libmolgrid/metal_context.h)
endif()

set ( LIBMOLGRID_HEADERS ${LIBMOLGRID_HEADERS}  PARENT_SCOPE)

if(BUILD_SHARED)
    add_library(libmolgrid_shared SHARED ${LIBMOLGRID_HEADERS} ${LIBMOLGRID_SOURCES})
    if(LIBMOLGRID_USE_CUDA)
      SET_TARGET_PROPERTIES(libmolgrid_shared PROPERTIES OUTPUT_NAME molgrid CUDA_SEPARABLE_COMPILATION OFF)
    else()
      SET_TARGET_PROPERTIES(libmolgrid_shared PROPERTIES OUTPUT_NAME molgrid)
    endif()
    target_link_libraries(libmolgrid_shared
        ${OPENBABEL3_LIBRARIES}
        ${Boost_LIBRARIES}
        ${ZLIB_LIBRARIES}
        ${CUDA_LIBRARIES}
        ${METAL_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
    )
    install(TARGETS libmolgrid_shared DESTINATION lib)
endif()

if(BUILD_STATIC)
    add_library(libmolgrid_static STATIC ${LIBMOLGRID_HEADERS} ${LIBMOLGRID_SOURCES})
    if(LIBMOLGRID_USE_CUDA)
      SET_TARGET_PROPERTIES(libmolgrid_static PROPERTIES OUTPUT_NAME molgrid CUDA_SEPARABLE_COMPILATION OFF POSITION_INDEPENDENT_CODE ON)
    else()
      SET_TARGET_PROPERTIES(libmolgrid_static PROPERTIES OUTPUT_NAME molgrid POSITION_INDEPENDENT_CODE ON)
    endif()
    target_link_libraries(libmolgrid_static
        ${OPENBABEL3_LIBRARIES}
        ${Boost_LIBRARIES}
        ${ZLIB_LIBRARIES}
        ${CUDA_LIBRARIES}
        ${METAL_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
    )
    install(TARGETS libmolgrid_static DESTINATION lib)
endif()

if(LIBMOLGRID_USE_METAL)
  find_program(XCRUN xcrun REQUIRED)

  set(METAL_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/metal/shaders.metal)
  set(METAL_AIR ${CMAKE_CURRENT_BINARY_DIR}/shaders.air)
  set(METAL_LIBRARY_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libmolgrid_shaders.metallib)

  if(BUILD_SHARED)
    target_compile_definitions(libmolgrid_shared PRIVATE
        METAL_LIBRARY_PATH="${METAL_LIBRARY_OUTPUT}")
  endif()
  if(BUILD_STATIC)
    target_compile_definitions(libmolgrid_static PRIVATE
        METAL_LIBRARY_PATH="${METAL_LIBRARY_OUTPUT}")
  endif()

  add_custom_command(
      OUTPUT ${METAL_AIR}
      COMMAND ${XCRUN} -sdk macosx metal
              -std=metal3.0
              -O2
              -c ${METAL_SOURCE}
              -o ${METAL_AIR}
      DEPENDS ${METAL_SOURCE}
      COMMENT "Compiling Metal shaders to AIR"
  )

  add_custom_command(
      OUTPUT ${METAL_LIBRARY_OUTPUT}
      COMMAND ${XCRUN} -sdk macosx metallib ${METAL_AIR} -o ${METAL_LIBRARY_OUTPUT}
      DEPENDS ${METAL_AIR}
      COMMENT "Linking Metal AIR to metallib"
  )

  add_custom_target(metal_shaders ALL DEPENDS ${METAL_LIBRARY_OUTPUT})

  if(BUILD_SHARED)
      add_dependencies(libmolgrid_shared metal_shaders)
  endif()
  if(BUILD_STATIC)
      add_dependencies(libmolgrid_static metal_shaders)
  endif()

  install(FILES ${METAL_LIBRARY_OUTPUT} DESTINATION lib)
endif()

install(FILES ${LIBMOLGRID_HEADERS} ${PROJECT_BINARY_DIR}/include/libmolgrid/config.h
        DESTINATION include/libmolgrid)
