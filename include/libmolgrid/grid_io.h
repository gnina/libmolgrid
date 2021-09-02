/** \file grid_io.h
 *
 *  Input/output routines for 3D grids.
 */

#ifndef GRID_IO_H_
#define GRID_IO_H_


#include "libmolgrid/cartesian_grid.h"
#include <iostream>

namespace libmolgrid {

// Docstring_read_dx
/* \brief Read in dx formatted grid and return initialized grid */
template <typename DType>
CartesianGrid<ManagedGrid<DType, 3> > read_dx(std::istream& in);
template <typename DType>
CartesianGrid<ManagedGrid<DType, 3> > read_dx(const std::string& fname);

/// Read dx formatted grid into provided grid, which must have the correct dimensions.
template <typename Dtype>
void read_dx(std::istream& in, Grid<Dtype, 3>& grid);
template <typename Dtype>
void read_dx(const std::string& fname, Grid<Dtype, 3>& grid);

///read in binary file, grid must be correct size
template <class G>
void read_bin(std::istream& in, G& grid) {
    in.read((char*)grid.data(), grid.size() * sizeof(typename G::type));
}


//output routines

// Docstring_write_dx
/** \brief output grid as dx formatted file
 * Values are multiplied by scale, which may be necessary to adjust for limited precision in the text-based format
 */
template <typename DType>
void write_dx(std::ostream& out, const Grid<DType, 3>& grid, const float3& center, float resolution, float scale=1.0);
template <typename DType>
void write_dx(const std::string& fname, const Grid<DType, 3>& grid, const float3& center, float resolution, float scale=1.0);

// Docstring_write_dx_grids
/** \brief Output multiple grids using type names as a suffix.
 * @param[in] prefix filename will have form [prefix]_[typename].dx
 * @param[in] names must have same size as first dimension of grid
 * @param[in] grid input grids
 * @param[in] center
 * @param[in] resolution
 * @param[in] scale multiply each value by this factor
 */
template <typename Dtype>
void write_dx_grids(const std::string& prefix, const std::vector<std::string>& names, const Grid<Dtype, 4>& grid, const float3& center, float resolution, float scale=1.0);

// Docstring_read_dx_grids
/** \brief Read multiple grids using type names as a suffix.  Grids must be correctly sized
 * @param[in] prefix filename will have form [prefix]_[typename].dx
 * @param[in] names must have same size as first dimension of grid
 * @param[in] grid input grids
 */
template <typename Dtype>
void read_dx_grids(const std::string& prefix, const std::vector<std::string>& names, Grid<Dtype, 4>& grid);

// Docstring_write_map
/// output grid as autodock map formatted file
template <typename DType>
void write_map(std::ostream& out, const Grid<DType, 3>& grid, const float3& center, float resolution, float scale=1.0);
template <typename DType>
void write_map(const std::string& fname, const Grid<DType, 3>& grid, const float3& center, float resolution, float scale=1.0);

//dump raw data in binary
template <class G>
void write_bin(std::ostream& out, const G& grid) {
    out.write((char*)grid.data(), grid.size() * sizeof(typename G::type));
}

}



#endif /* GRID_IO_H_ */
