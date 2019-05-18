/*
 * coordinateset.cpp
 *
 *  Created on: Mar 22, 2019
 *      Author: dkoes
 */
#include <cuda_runtime.h>

#include "libmolgrid/coordinateset.h"
#include "libmolgrid/atom_typer.h"
#include <openbabel/obiter.h>

namespace libmolgrid {

using namespace std;
using namespace OpenBabel;

CoordinateSet::CoordinateSet(OBMol *mol): CoordinateSet(mol, defaultGninaLigandTyper) {}

//initialize with obmol
CoordinateSet::CoordinateSet(OBMol *mol, const AtomTyper& typer)
    : max_type(typer.num_types()) {

  vector<float3> c; c.reserve(mol->NumAtoms());
  vector<float> types;  types.reserve(mol->NumAtoms());
  vector<vector<float> > vector_types;  vector_types.reserve(mol->NumAtoms());
  vector<float> radii; radii.reserve(mol->NumAtoms());
  vector<float> vec;

  FOR_ATOMS_OF_MOL(a, mol){
    OBAtom *atom = &*a; //convert from iterator

    if(typer.is_vector_typer()) {

      float radius = typer.get_atom_type_vector(atom, vec);
      if(radius > 0) { //don't ignore
        c.push_back(make_float3(atom->GetX(), atom->GetY(), atom->GetZ()));
        vector_types.push_back(vec);
        radii.push_back(radius);
      }

    } else {
      auto type_rad = typer.get_atom_type_index(atom);
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
  memcpy(coord.cpu().data(), &c[0], sizeof(float3)*N);

  radius = MGrid1f(N);
  memcpy(radius.cpu().data(), &radii[0], sizeof(float)*N);

  if(typer.is_vector_typer()) {
    type_vector = MGrid2f(N,max_type);
    memcpy(type_vector.cpu().data(), &vector_types[0], sizeof(float)*N*max_type);
  } else {
    type_index = MGrid1f(N);
    memcpy(type_index.cpu().data(), &types[0], sizeof(float)*N);
  }
}

//initialize with indexed types
CoordinateSet::CoordinateSet(const std::vector<float3>& c, const std::vector<int>& t, const std::vector<float>& r, unsigned maxt):
  coord(c.size(),3), type_index(c.size()), radius(c.size()), max_type(maxt) {
  unsigned N = c.size();
  if(N != t.size()) {
    throw std::invalid_argument("Types and coordinates are of different sizes");
  }
  if(N != r.size()) {
    throw std::invalid_argument("Radii and coordinates are of different sizes");
  }

  //copy data
  type_index.tocpu(); radius.tocpu(); coord.tocpu();
  memcpy(radius.cpu().data(), &r[0], sizeof(float)*N);
  assert(sizeof(float3)*N == sizeof(float)*coord.size());
  memcpy(coord.cpu().data(), &c[0], sizeof(float3)*N);

  //convert to float
  for(unsigned i = 0; i < N; i++) {
    type_index[i] = t[i];
  }
}

//initialize with indexed types (float)
CoordinateSet::CoordinateSet(const std::vector<float3>& c, const std::vector<float>& t, const std::vector<float>& r, unsigned maxt):
  coord(c.size(),3), type_index(c.size()), radius(c.size()), max_type(maxt) {
  unsigned N = c.size();
  if(N != t.size()) {
    throw std::invalid_argument("Types and coordinates are of different sizes");
  }
  if(N != r.size()) {
    throw std::invalid_argument("Radii and coordinates are of different sizes");
  }

  //copy data
  type_index.tocpu(); radius.tocpu(); coord.tocpu();
  memcpy(type_index.cpu().data(), &t[0], sizeof(float)*N);
  memcpy(radius.cpu().data(), &r[0], sizeof(float)*N);
  assert(sizeof(float3)*N == sizeof(float)*coord.size());
  memcpy(coord.cpu().data(), &c[0], sizeof(float3)*N);

}


///initialize with indexed types using grids - data is copied into coordinate set
CoordinateSet::CoordinateSet(const Grid2f& c, const Grid1f& t, const Grid1f& r, unsigned maxt):
    coord(c.dimension(0), c.dimension(1)), type_index(t.dimension(0)), radius(r.dimension(0)), max_type(maxt) {
  coord.copyFrom(c);
  type_index.copyFrom(t);
  radius.copyFrom(r);
}

CoordinateSet::CoordinateSet(const Grid2fCUDA& c, const Grid1fCUDA& t, const Grid1fCUDA& r, unsigned maxt):
    coord(c.dimension(0), c.dimension(1)), type_index(t.dimension(0)), radius(r.dimension(0)), max_type(maxt) {
  coord.copyFrom(c);
  type_index.copyFrom(t);
  radius.copyFrom(r);
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
  type_index.tocpu(); radius.tocpu(); coord.tocpu();
  memcpy(radius.cpu().data(), &r[0], sizeof(float)*N);
  assert(sizeof(float3)*N == sizeof(float)*coord.size());
  memcpy(coord.cpu().data(), &c[0], sizeof(float3)*N);
  memcpy(type_vector.cpu().data(), &t[0], sizeof(float)*N*max_type);

}

//vector types in grids
CoordinateSet::CoordinateSet(const Grid2f& c, const Grid2f& t, const Grid1f& r):
    coord(c.dimension(0),c.dimension(1)), type_vector(t.dimension(0), t.dimension(1)), radius(r.dimension(0)), max_type(t.dimension(1)) {
  coord.copyFrom(c);
  type_vector.copyFrom(t);
  radius.copyFrom(r);
}

CoordinateSet::CoordinateSet(const Grid2fCUDA& c, const Grid2fCUDA& t, const Grid1fCUDA& r):
    coord(c.dimension(0),c.dimension(1)), type_vector(t.dimension(0), t.dimension(1)), radius(r.dimension(0)), max_type(t.dimension(1)) {
  coord.copyFrom(c);
  type_vector.copyFrom(t);
  radius.copyFrom(r);
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

float3 CoordinateSet::center() const {
  float3 ret = make_float3(0,0,0);
  unsigned N = coord.dimension(0);

  coord.tocpu();
  if(N == 0) return ret;
  for(unsigned i = 0; i < N; i++) {
    ret.x += coord(i,0);
    ret.y += coord(i,1);
    ret.z += coord(i,2);
  }
  ret.x /= N;
  ret.y /= N;
  ret.z /= N;
  return ret;
}

void CoordinateSet::dump(std::ostream& out) const {
  unsigned N = coord.dimension(0);
  coord.tocpu();
  if(N == 0) return;
  for(unsigned i = 0; i < N; i++) {
    out << coord(i,0) << "," << coord(i,1) << "," << coord(i,2);
    if(has_indexed_types()) {
      out << " " << type_index(i);
    }
    //todo vector types
    out << "\n";
  }
}

void CoordinateSet::size_like(const CoordinateSet& s) {
  coord = coord.resized(s.coord.dimension(0), 3);
  type_index = type_index.resized(s.type_index.dimension(0));
  type_vector = type_vector.resized(s.type_vector.dimension(0), s.type_vector.dimension(1));
  radius = radius.resized(s.radius.dimension(0));
}

void CoordinateSet::copyInto(const CoordinateSet& s) {
  size_like(s);
  coord.copyFrom(s.coord);
  type_index.copyFrom(s.type_index);
  type_vector.copyFrom(s.type_vector);
  radius.copyFrom(s.radius);

  max_type = s.max_type;
  src = s.src;
}



}
