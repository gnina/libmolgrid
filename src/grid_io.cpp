/*
 * grid_io.cpp
 *
 *  Created on: Apr 19, 2019
 *      Author: dkoes
 */



#include "libmolgrid/grid_io.h"
#include <boost/algorithm/string.hpp>
#include <fstream>
#include <iomanip>

namespace libmolgrid {

using namespace std;
using namespace boost;

template <typename DType>
CartesianGrid<ManagedGrid<DType, 3> > read_dx(std::istream& in) {
  string line;
  vector<string> tokens;

  float res = 0;
  getline(in, line);
  split(tokens, line, is_any_of(" \t"), token_compress_on);
  if (tokens.size() != 8) throw invalid_argument("Could not read dx file: tokens != 8");
  unsigned n = lexical_cast<unsigned>(tokens[7]);
  if (lexical_cast<unsigned>(tokens[6]) != n) throw invalid_argument("Could not read dx file: tokens[6] != "+itoa(n));
  if (lexical_cast<unsigned>(tokens[5]) != n) throw invalid_argument("Could not read dx file: tokens[5] != "+itoa(n));

  //the center
  getline(in, line);
  split(tokens, line, is_any_of(" \t"), token_compress_on);
  if (tokens.size() != 4) throw invalid_argument("Could not read dx file: tokens != 4");
  double x = lexical_cast<double>(tokens[1]);
  double y = lexical_cast<double>(tokens[2]);
  double z = lexical_cast<double>(tokens[3]);

  //the transformation matrix, which has the resolution
  getline(in, line);
  split(tokens, line, is_any_of(" \t"), token_compress_on);
  if (tokens.size() != 4) throw invalid_argument("Could not read dx file: tokens != 4 (2)");
  res = lexical_cast<float>(tokens[1]);

  getline(in, line);
  split(tokens, line, is_any_of(" \t"), token_compress_on);
  if (tokens.size() != 4) throw invalid_argument("Could not read dx file: tokens != 4 (3)");
  if (res != lexical_cast<float>(tokens[2])) throw invalid_argument("Could not read dx file: mismatch");

  getline(in, line);
  split(tokens, line, is_any_of(" \t"), token_compress_on);
  if (tokens.size() != 4) throw invalid_argument("Could not read dx file: tokens != 4 (4)");;
  if (res != lexical_cast<float>(tokens[3])) throw invalid_argument("Could not read dx file: mismatch (2)");;

  //figure out center
  double half = res * n / 2.0;
  float3 center;
  center.x = x + half;
  center.y = y + half;
  center.z = z + half;

  //grid connections
  getline(in, line);
  //object 3
  getline(in, line);

  //data begins
  ManagedGrid<DType, 3> grid(n,n,n);

  unsigned total = 0;
  for (unsigned i = 0; i < n; i++) {
    for (unsigned j = 0; j < n; j++) {
      for (unsigned k = 0; k < n; k++) {
        in >> grid[i][j][k];
        total++;
      }
    }
  }
  if (total != n * n * n) throw invalid_argument("Could not read dx file: incorrect number of data points ("+itoa(total)+" vs "+itoa(n*n*n));

  return CartesianGrid<ManagedGrid<DType, 3> >(grid, center, res);
}

///read dx grid from file name
template <typename DType>
CartesianGrid<ManagedGrid<DType, 3> > read_dx(const std::string& fname) {
  std::ifstream f(fname.c_str());
  if(!f) throw invalid_argument("Could not read file "+fname);
  return read_dx<DType>(f);
}


template <typename DType>
void write_dx(std::ostream& out, const Grid<DType, 3>& grid, const float3& center, float resolution) {
  unsigned n = grid.dimension(0);
  out.precision(5);
  setprecision(5);
  out << fixed;
  out << "object 1 class gridpositions counts " << n << " " << n << " " << " "
      << n << "\n";

  // figure out origin from center and dim/res
  double half = resolution * (n-1) / 2.0;
  out << "origin " << center.x-half << " " << center.y-half << " " << center.z-half << "\n";

  out << "delta " << resolution << " 0 0\ndelta 0 " << resolution
      << " 0\ndelta 0 0 " << resolution << "\n";
  out << "object 2 class gridconnections counts " << n << " " << n << " " << " "
      << n << "\n";
  out << "object 3 class array type double rank 0 items [ " << n * n * n
      << "] data follows\n";
  //now coordinates - x,y,z
  unsigned total = 0;
  for (unsigned i = 0; i < n; i++) {
    for (unsigned j = 0; j < n; j++) {
      for (unsigned k = 0; k < n; k++) {
        out << grid[i][j][k];
        total++;
        if (total % 3 == 0)
          out << "\n";
        else
          out << " ";
      }
    }
  }
}

///output dx to file name
template <typename DType>
void write_dx(const std::string& fname, const Grid<DType, 3>& grid, const float3& center, float resolution) {
  std::ofstream f(fname.c_str());
  if(!f) throw invalid_argument("Could not open file "+fname);
  return write_dx(f, grid, center, resolution);
}

///output autodock4 to stream
template <typename DType>
void write_map(std::ostream& out, const Grid<DType, 3>& grid, const float3& center, float resolution) {
  unsigned max = grid.dimension(0);
  out.precision(5);
  out << "GRID_PARAMETER_FILE\nGRID_DATA_FILE\nMACROMOLECULE\n";
  out << "SPACING " << resolution << "\n";
  out << "NELEMENTS " << max - 1 << " " << max - 1 << " " << max - 1 << "\n";
  out << "CENTER " << center.x << " " << center.y << " " << center.z << "\n";

  //now coordinates - z,y,x
  for (unsigned k = 0; k < max; k++) {
    for (unsigned j = 0; j < max; j++) {
      for (unsigned i = 0; i < max; i++) {
        out << grid[i][j][k] << "\n";
      }
    }
  }
}

///output autodock4 to filename
template <typename DType>
void write_map(const std::string& fname, const Grid<DType, 3>& grid, const float3& center, float resolution) {
  std::ofstream f(fname.c_str());
  if(!f) throw invalid_argument("Could not open file "+fname);
  return write_map(f, grid, center, resolution);
}



template CartesianGrid<ManagedGrid<float, 3> > read_dx(std::istream& in);
template CartesianGrid<ManagedGrid<double, 3> > read_dx(std::istream& in);
template CartesianGrid<ManagedGrid<float, 3> > read_dx(const std::string& fname);
template CartesianGrid<ManagedGrid<double, 3> > read_dx(const std::string& fname);

template void write_dx(std::ostream& out, const Grid<float, 3>& grid, const float3& center, float resolution);
template void write_dx(const std::string& fname, const Grid<float, 3>& grid, const float3& center, float resolution);
template void write_dx(std::ostream& out, const Grid<double, 3>& grid, const float3& center, float resolution);
template void write_dx(const std::string& fname, const Grid<double, 3>& grid, const float3& center, float resolution);

template void write_map(std::ostream& out, const Grid<float, 3>& grid, const float3& center, float resolution);
template void write_map(const std::string& fname, const Grid<float, 3>& grid, const float3& center, float resolution);
template void write_map(std::ostream& out, const Grid<double, 3>& grid, const float3& center, float resolution);
template void write_map(const std::string& fname, const Grid<double, 3>& grid, const float3& center, float resolution);


}
