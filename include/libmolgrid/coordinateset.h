/** \file coordinateset.h
 * Class for holding typed atomic coordinates.
 */
/*
 *  Created on: Mar 22, 2019
 *      Author: dkoes
 */

#ifndef COORDINATESET_H_
#define COORDINATESET_H_


#include <vector>
#include <openbabel/mol.h>
#include "libmolgrid/managed_grid.h"

namespace libmolgrid {

class AtomTyper;

/** \brief A collection of typed atomic coordinates
 *
 * Types may be specified either as an index or a dense vector.
 * Typically, only one type formated will be initialized although
 * a vector one-hot encoding of an index type can be created.
 *
 */
struct CoordinateSet {
  MGrid2f coord{0,0};
  MGrid1f type_index{0}; //this should be integer
  MGrid2f type_vector{0,0};
  MGrid1f radius{0};
  unsigned max_type = 0;  //for indexed types, non-inclusive max
  const char *src = nullptr; //mostly for debugging, source of coordinates

  CoordinateSet() {}

  ///initialize with obmol
  CoordinateSet(OpenBabel::OBMol *mol, const AtomTyper& typer);
  CoordinateSet(OpenBabel::OBMol *mol); //use defaultGninaLigandTypes

  ///initialize with indexed types
  CoordinateSet(const std::vector<float3>& c, const std::vector<unsigned>& t, const std::vector<float>& r, unsigned maxt);
  CoordinateSet(const std::vector<float3>& c, const std::vector<float>& t, const std::vector<float>& r, unsigned maxt);

  ///initialize with indexed types using grids - data is copied into coordinate set
  CoordinateSet(const Grid2f& c, const Grid1f& t, const Grid1f& r, unsigned maxt);
  CoordinateSet(const Grid2fCUDA& c, const Grid1fCUDA& t, const Grid1fCUDA& r, unsigned maxt);

  ///initialize with vector types
  CoordinateSet(const std::vector<float3>& c, const std::vector<std::vector<float> >& t, const std::vector<float>& r);

  ///initialize with vector types using grids - data is copied into coordinate set
  CoordinateSet(const Grid2f& c, const Grid2f& t, const Grid1f& r);
  CoordinateSet(const Grid2fCUDA& c, const Grid2fCUDA& t, const Grid1fCUDA& r);

  /// return true if index types are available
  bool has_indexed_types() const { return type_index.size() > 0; }

  /// return true if vector types are available
  bool has_vector_types() const { return type_vector.size() > 0; }

  ///convert index types to vector types in-place
  void make_vector_types();

  unsigned num_types() const { return max_type; }

  ///number of atoms
  unsigned size() const { return coord.dimension(0); }

  ///return mean of coordinates
  float3 center() const;

  void togpu() { coord.togpu(); type_index.togpu(); type_vector.togpu(); radius.togpu(); }
  void tocpu() { coord.tocpu(); type_index.tocpu(); type_vector.tocpu(); radius.tocpu(); }

  //test for pointer equality, not particularly useful, but needed by boost::python
  bool operator==(const CoordinateSet& rhs) const {
    return max_type == rhs.max_type && coord == rhs.coord && type_index == rhs.type_index && type_vector == rhs.type_vector && radius == rhs.radius;
  }

  ///return deep copy
  CoordinateSet clone() const {
    CoordinateSet ret(*this);
    ret.coord = coord.clone();
    ret.radius = radius.clone();
    ret.type_index = type_index.clone();
    ret.type_vector = type_vector.clone();
    return ret;
  }

  ///for debugging
  void dump(std::ostream& out) const;
};

}


#endif /* COORDINATESET_H_ */
