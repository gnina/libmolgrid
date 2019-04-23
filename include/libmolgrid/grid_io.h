/** \file grid_io.h
 *
 *  Input/output routines for 3D grids.
 */

#ifndef GRID_IO_H_
#define GRID_IO_H_


#include "libmolgrid/cartesian_grid.h"
#include <iostream>

namespace libmolgrid {

/// Read in dx formatted grid and return initialized grid
template <typename DType>
CartesianGrid<ManagedGrid<DType, 3> > read_dx(std::istream& in);
template <typename DType>
CartesianGrid<ManagedGrid<DType, 3> > read_dx(const std::string& fname);

///read in binary file, grid must be correct size
template <class G>
void read_bin(std::istream& in, G& grid) {
    in.read((char*)grid.data(), grid.size() * sizeof(typename G::type));
}


//output routines

/// output grid as dx formatted file
template <typename DType>
void write_dx(std::ostream& out, const Grid<DType, 3>& grid, const float3& center, float resolution);
template <typename DType>
void write_dx(const std::string& fname, const Grid<DType, 3>& grid, const float3& center, float resolution);

/// output grid as dx formatted file
template <typename DType>
void write_dx(std::ostream& out, const Grid<DType, 3>& grid, const float3& center, float resolution);
template <typename DType>
void write_dx(const std::string& fname, const Grid<DType, 3>& grid, const float3& center, float resolution);

/// output grid as autodock map formatted file
template <typename DType>
void write_map(std::ostream& out, const Grid<DType, 3>& grid, const float3& center, float resolution);
template <typename DType>
void write_map(const std::string& fname, const Grid<DType, 3>& grid, const float3& center, float resolution);

//dump raw data in binary
template <class G>
void write_bin(std::ostream& out, const G& grid) {
    out.write((char*)grid.data(), grid.size() * sizeof(typename G::type));
}


extern template CartesianGrid<ManagedGrid<float, 3> > read_dx(std::istream& in);
extern template CartesianGrid<ManagedGrid<double, 3> > read_dx(std::istream& in);
extern template CartesianGrid<ManagedGrid<float, 3> > read_dx(const std::string& fname);
extern template CartesianGrid<ManagedGrid<double, 3> > read_dx(const std::string& fname);

extern template void write_dx(std::ostream& out, const Grid<float, 3>& grid, const float3& center, float resolution);
extern template void write_dx(const std::string& fname, const Grid<float, 3>& grid, const float3& center, float resolution);
extern template void write_dx(std::ostream& out, const Grid<double, 3>& grid, const float3& center, float resolution);
extern template void write_dx(const std::string& fname, const Grid<double, 3>& grid, const float3& center, float resolution);

extern template void write_map(std::ostream& out, const Grid<float, 3>& grid, const float3& center, float resolution);
extern template void write_map(const std::string& fname, const Grid<float, 3>& grid, const float3& center, float resolution);
extern template void write_map(std::ostream& out, const Grid<double, 3>& grid, const float3& center, float resolution);
extern template void write_map(const std::string& fname, const Grid<double, 3>& grid, const float3& center, float resolution);
}



#endif /* GRID_IO_H_ */
