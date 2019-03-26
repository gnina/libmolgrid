/*
 * coordinateset.cpp
 *
 *  Created on: Mar 22, 2019
 *      Author: dkoes
 */
#include <cuda_runtime.h>

#include "coordinateset.h"
#include "atom_typer.h"

namespace libmolgrid {

using namespace std;
using namespace OpenBabel;

//initialize with obmol
CoordinateSet::CoordinateSet(OBMol *mol, AtomTyper& typer)
    : max_type(typer.num_types()) {

  vector<float3> c; c.reserve(mol->NumAtoms());
  vector<float> types;  types.reserve(mol->NumAtoms());
  vector<vector<float> > vector_types;  vector_types.reserve(mol->NumAtoms());
  vector<float> radii; radii.reserve(mol->NumAtoms());
  vector<float> vec;

  FOR_ATOMS_OF_MOL(a, mol){
    OBAtom *atom = &*a; //convert from iterator

    if(typer.is_vector_typer()) {

      float radius = typer.get_atom_type(atom, vec);
      if(radius > 0) { //don't ignore
        c.push_back(make_float3(atom->GetX(), atom->GetY(), atom->GetZ()));
        vector_types.push_back(vec);
        radii.push_back(radius);
      }

    } else {
      auto type_rad = typer.get_atom_type(atom);
      int type = type_rad.first;
      float r = type_rad.second;
      if(type >= (int)max_type) throw invalid_argument("Invalid type");
      if(type >= 0) { //don't ignore atom
        c.push_back(make_float3(atom->GetX(), atom->GetY(), atom->GetZ()));
        types.push_back(type);
        radii.push_back(r);
      }
    }
  }

  //allocate grids and initialize
  unsigned N = c.size();
  coord = MGrid2f(N,3);
  assert(sizeof(float3)*N == sizeof(float)*coord.size());
  memcpy(coord.pointer().get(), &c[0], sizeof(float3)*N);

  radius = MGrid1f(N);
  memcpy(radius.pointer().get(), &radii[0], sizeof(float)*N);

  if(typer.is_vector_typer()) {
    type_vector = MGrid2f(N,max_type);
    memcpy(type_vector.pointer().get(), &vector_types[0], sizeof(float)*N*max_type);
  } else {
    type_index = MGrid1f(N);
    memcpy(type_index.pointer().get(), &types[0], sizeof(float)*N*max_type);
  }
}

//initialize with indexed types
CoordinateSet::CoordinateSet(const std::vector<float3>& c, const std::vector<unsigned>& t, const std::vector<float>& r, unsigned maxt):
  coord(c.size(),3), type_index(c.size()), radius(c.size()), max_type(maxt) {
  unsigned N = c.size();
  if(N != t.size()) {
    throw std::invalid_argument("Types and coordinates are of different sizes");
  }
  if(N != r.size()) {
    throw std::invalid_argument("Radii and coordinates are of different sizes");
  }

  //copy data
  memcpy(radius.pointer().get(), &r[0], sizeof(float)*N);
  assert(sizeof(float3)*N == sizeof(float)*coord.size());
  memcpy(coord.pointer().get(), &c[0], sizeof(float3)*N);

  //convert to float
  for(unsigned i = 0; i < N; i++) {
    type_index[i] = t[i];
  }
}

inline size_t typ_vec_size(const std::vector<std::vector<float> >& t) {
  if(t.size() == 0) return 0;
  return t[0].size();
}

//initialize with vector types
CoordinateSet::CoordinateSet(const std::vector<float3>& c, const std::vector<std::vector<float> >& t, const std::vector<float>& r):
  coord(c.size(),3), type_vector(c.size(),typ_vec_size(t)), radius(c.size()), max_type(typ_vec_size(t)) {

  unsigned N = c.size();
  if(N != t.size()) {
    throw std::invalid_argument("Types and coordinates are of different sizes");
  }
  if(N != r.size()) {
    throw std::invalid_argument("Radii and coordinates are of different sizes");
  }

  //copy data
  memcpy(radius.pointer().get(), &r[0], sizeof(float)*N);
  assert(sizeof(float3)*N == sizeof(float)*coord.size());
  memcpy(coord.pointer().get(), &c[0], sizeof(float3)*N);
  memcpy(type_vector.pointer().get(), &t[0], sizeof(float)*N*max_type);

}

///convert index types to vector types in-place
void CoordinateSet::make_vector_types() {
  unsigned N = type_index.size();
  type_vector = MGrid2f(N, max_type); //grid are always zero initialized
  for(unsigned i = 0; i < N; i++) {
    unsigned t = type_index[i];
    if(t < max_type) {
      type_vector[i][t] = 1.0;
    }
  }

}

}
