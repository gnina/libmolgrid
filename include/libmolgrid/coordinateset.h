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
  MGrid2f coords{0,3}; //coordinats
  MGrid1f type_index{0}; //this should be integer
  MGrid2f type_vector{0,0};
  MGrid1f radii{0}; ///radii - for type_index, indexed by atom, for type vector, indexed by type
  unsigned max_type = 0;  //for indexed types, non-inclusive max
  const char *src = nullptr; //mostly for debugging, source of coordinates

  CoordinateSet() {}

  ///initialize with obmol
  CoordinateSet(OpenBabel::OBMol *mol, const AtomTyper& typer);
  CoordinateSet(OpenBabel::OBMol *mol); //use defaultGninaLigandTypes

  ///initialize with indexed types
  CoordinateSet(const std::vector<float3>& c, const std::vector<int>& t, const std::vector<float>& r, unsigned maxt);
  CoordinateSet(const std::vector<float3>& c, const std::vector<float>& t, const std::vector<float>& r, unsigned maxt);

  ///initialize with indexed types using grids - data is copied into coordinate set
  CoordinateSet(const Grid2f& coords, const Grid1f& t, const Grid1f& radii, unsigned maxt);
  CoordinateSet(const Grid2fCUDA& coords, const Grid1fCUDA& t, const Grid1fCUDA& radii, unsigned maxt);

  ///initialize with vector types
  CoordinateSet(const std::vector<float3>& c, const std::vector<std::vector<float> >& t, const std::vector<float>& r);

  ///initialize with vector types using grids - data is copied into coordinate set
  CoordinateSet(const Grid2f& cr, const Grid2f& t, const Grid1f& r);
  CoordinateSet(const Grid2fCUDA& cr, const Grid2fCUDA& t, const Grid1fCUDA& r);

  /// return true if index types are available
  bool has_indexed_types() const { return type_index.size() > 0 || type_vector.size() == 0; }

  /// return true if vector types are available
  bool has_vector_types() const { return type_vector.size() > 0; }

  ///convert index types to vector types in-place
  void make_vector_types();

  unsigned num_types() const { return max_type; }
  void set_num_types(unsigned maxt) { max_type = maxt; }

  ///number of atoms
  unsigned size() const { return coords.dimension(0); }

  ///return mean of coordinates
  float3 center() const;

  void togpu(bool copy=true) { coords.togpu(copy); type_index.togpu(copy); type_vector.togpu(copy); radii.togpu(copy);}
  void tocpu(bool copy=true) { coords.tocpu(copy); type_index.tocpu(copy); type_vector.tocpu(copy); radii.tocpu(copy);}

  //test for pointer equality, not particularly useful, but needed by boost::python
  bool operator==(const CoordinateSet& rhs) const {
    return max_type == rhs.max_type && coords == rhs.coords && type_index == rhs.type_index
        && type_vector == rhs.type_vector && radii == rhs.radii;
  }

  ///return deep copy
  CoordinateSet clone() const {
    CoordinateSet ret(*this);
    ret.coords = coords.clone();
    ret.type_index = type_index.clone();
    ret.type_vector = type_vector.clone();
    ret.radii = radii.clone();
    return ret;
  }

  /// size this to have the same size as s without copying data
  void size_like(const CoordinateSet& s); 

  /// copy contents of src into this, attempting to avoid reallocation if possible
  void copyInto(const CoordinateSet& src);
  ///for debugging
  void dump(std::ostream& out) const;
};

}


#endif /* COORDINATESET_H_ */
