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
#include "atom_typer.h"
#include "managed_grid.h"

namespace libmolgrid {

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
  const unsigned max_type = 0;  //for indexed types, non-inclusive max

  ///initialize with obmol
  CoordinateSet(OpenBabel::OBMol *mol, AtomTyper& typer);

  ///initialize with indexed types
  CoordinateSet(const std::vector<float3>& c, const std::vector<unsigned>& t, const std::vector<float>& r, unsigned maxt);

  ///initialize with vector types
  CoordinateSet(const std::vector<float3>& c, const std::vector<std::vector<float> >& t, const std::vector<float>& r);

  /// return true if index types are available
  bool has_indexed_types() const { return type_index.size() > 0; }

  /// return true if vector types are available
  bool has_vector_types() const { return type_vector.size() > 0; }

  ///convert index types to vector types in-place
  void make_vector_types();

  unsigned num_types() const { return max_type; }

};

}


#endif /* COORDINATESET_H_ */
