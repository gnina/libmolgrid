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

//parse dx header
unsigned read_dx_helper(std::istream& in, float3& center, float& res) {
  string line;
  vector<string> tokens;

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
  double half = res * (n-1) / 2.0;
  center.x = x + half;
  center.y = y + half;
  center.z = z + half;

  //grid connections
  getline(in, line);
  //object 3
  getline(in, line);
  return n;
}

template <typename DType>
CartesianGrid<ManagedGrid<DType, 3> > read_dx(std::istream& in) {

  float3 center;
  float res;
  unsigned n = read_dx_helper(in, center, res);
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

template <typename Dtype>
void read_dx(std::istream& in, Grid<Dtype, 3>& grid) {
  float3 center;
  float res;
  unsigned n = read_dx_helper(in, center, res);
  if(n != grid.dimension(0)) throw invalid_argument("Grid incorrect size in read_dx: "+itoa(n) +" != " +itoa(grid.dimension(0)));
  if(n != grid.dimension(1)) throw invalid_argument("Grid incorrect size in read_dx: "+itoa(n) +" != " +itoa(grid.dimension(1)));
  if(n != grid.dimension(2)) throw invalid_argument("Grid incorrect size in read_dx: "+itoa(n) +" != " +itoa(grid.dimension(2)));

  //data begins
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

}

///read dx grid from file name
template <typename Dtype>
void read_dx(const std::string& fname, Grid<Dtype, 3>& grid) {
  std::ifstream f(fname.c_str());
  if(!f) throw invalid_argument("Could not read file "+fname);
  read_dx<Dtype>(f, grid);
}



template <typename DType>
void write_dx(std::ostream& out, const Grid<DType, 3>& grid, const float3& center, float resolution, float scale) {
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
        out << grid[i][j][k]*scale;
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
void write_dx(const std::string& fname, const Grid<DType, 3>& grid, const float3& center, float resolution, float scale) {
  std::ofstream f(fname.c_str());
  if(!f) throw invalid_argument("Could not open file "+fname);
  write_dx(f, grid, center, resolution, scale);
}

template <typename Dtype>
void write_dx_grids(const std::string& prefix, const std::vector<std::string>& names, const Grid<Dtype, 4>& grid,
    const float3& center, float resolution, float scale) {
  if(names.size() != grid.dimension(0))
    throw std::invalid_argument("Number of names and number of grids doesn't match in write_dx_grids: "+itoa(names.size())+ " != "+itoa(grid.dimension(0)));

  for(unsigned i = 0, n = names.size(); i < n; i++) {
    string fname = prefix+"_"+names[i]+".dx";
    if(fname.length() > 255) { //max file name length on linux
      fname = fname.substr(0,250) + ".dx";
    }
    write_dx(fname, grid[i], center, resolution, scale);
  }
}

template <typename Dtype>
void read_dx_grids(const std::string& prefix, const std::vector<std::string>& names, Grid<Dtype, 4>& grid) {
  if(names.size() != grid.dimension(0))
    throw std::invalid_argument("Number of names and number of grids doesn't match in read_dx_grids: "+itoa(names.size())+ " != "+itoa(grid.dimension(0)));

  for(unsigned i = 0, n = names.size(); i < n; i++) {
    string fname = prefix+"_"+names[i]+".dx";
    Grid<Dtype, 3> g = grid[i];
    read_dx<Dtype>(fname, g);
  }
}

///output autodock4 to stream
template <typename DType>
void write_map(std::ostream& out, const Grid<DType, 3>& grid, const float3& center, float resolution, float scale) {
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
        out << grid[i][j][k]*scale << "\n";
      }
    }
  }
}

///output autodock4 to filename
template <typename DType>
void write_map(const std::string& fname, const Grid<DType, 3>& grid, const float3& center, float resolution, float scale) {
  std::ofstream f(fname.c_str());
  if(!f) throw invalid_argument("Could not open file "+fname);
  write_map(f, grid, center, resolution, scale);
}



template CartesianGrid<ManagedGrid<float, 3> > read_dx(std::istream& in);
template CartesianGrid<ManagedGrid<double, 3> > read_dx(std::istream& in);
template CartesianGrid<ManagedGrid<float, 3> > read_dx(const std::string& fname);
template CartesianGrid<ManagedGrid<double, 3> > read_dx(const std::string& fname);

template void write_dx(std::ostream&, const Grid<float, 3>&, const float3&, float, float);
template void write_dx(const std::string&, const Grid<float, 3>&, const float3&, float, float);
template void write_dx(std::ostream&, const Grid<double, 3>&, const float3&, float, float);
template void write_dx(const std::string&, const Grid<double, 3>&, const float3&, float, float);

template void write_dx_grids(const std::string&, const std::vector<std::string>&, const Grid<float, 4>&, const float3&, float, float);
template void write_dx_grids(const std::string&, const std::vector<std::string>&, const Grid<double, 4>&, const float3&, float, float);

template void read_dx_grids(const std::string&, const std::vector<std::string>&, Grid<float, 4>&);
template void read_dx_grids(const std::string&, const std::vector<std::string>&, Grid<double, 4>&);

template void write_map(std::ostream&, const Grid<float, 3>&, const float3&, float, float);
template void write_map(const std::string&, const Grid<float, 3>&, const float3&, float, float);
template void write_map(std::ostream&, const Grid<double, 3>&, const float3&, float, float);
template void write_map(const std::string&, const Grid<double, 3>&, const float3&, float, float);


}
