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

  vector<float4> c; c.reserve(mol->NumAtoms());
  vector<float> types;  types.reserve(mol->NumAtoms());
  vector<vector<float> > vector_types;  vector_types.reserve(mol->NumAtoms());
  vector<float> radii; radii.reserve(mol->NumAtoms());
  vector<float> vec;

  FOR_ATOMS_OF_MOL(a, mol){
    OBAtom *atom = &*a; //convert from iterator

    if(typer.is_vector_typer()) {

      float radius = typer.get_atom_type_vector(atom, vec);
      if(radius > 0) { //don't ignore
        c.push_back(make_float4(atom->GetX(), atom->GetY(), atom->GetZ(), radius));
        vector_types.push_back(vec);
      }
    } else {
      auto type_rad = typer.get_atom_type_index(atom);
      int type = type_rad.first;
      float r = type_rad.second;
      if(type >= (int)max_type) throw invalid_argument("Invalid type");
      if(type >= 0) { //don't ignore atom
        c.push_back(make_float4(atom->GetX(), atom->GetY(), atom->GetZ(), r));
        types.push_back(type);
      }
    }
  }

  //allocate grids and initialize
  unsigned N = c.size();
  coord_radius = MGrid2f(N,4);
  assert(sizeof(float4)*N == sizeof(float)*coord_radius.size());
  memcpy(coord_radius.cpu().data(), &c[0], sizeof(float4)*N);

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
  coord_radius(c.size(),4), type_index(c.size()), max_type(maxt) {
  unsigned N = c.size();
  if(N != t.size()) {
    throw std::invalid_argument("Types and coordinates are of different sizes");
  }
  if(N != r.size()) {
    throw std::invalid_argument("Radii and coordinates are of different sizes");
  }

  //copy data
  type_index.tocpu(); coord_radius.tocpu();
  for(unsigned i = 0; i < N; i++) {
    type_index[i] = t[i];   //convert to float
    float3 p = c[i];
    coord_radius[i][0] = p.x;
    coord_radius[i][1] = p.y;
    coord_radius[i][2] = p.z;
    coord_radius[i][3] = r[i];
  }
}

//initialize with indexed types (float)
CoordinateSet::CoordinateSet(const std::vector<float3>& c, const std::vector<float>& t, const std::vector<float>& r, unsigned maxt):
  coord_radius(c.size(),4), type_index(c.size()), max_type(maxt) {
  unsigned N = c.size();
  if(N != t.size()) {
    throw std::invalid_argument("Types and coordinates are of different sizes");
  }
  if(N != r.size()) {
    throw std::invalid_argument("Radii and coordinates are of different sizes");
  }

  //copy data
  type_index.tocpu(); coord_radius.tocpu();
  memcpy(type_index.cpu().data(), &t[0], sizeof(float)*N);

  for(unsigned i = 0; i < N; i++) {
    float3 p = c[i];
    coord_radius[i][0] = p.x;
    coord_radius[i][1] = p.y;
    coord_radius[i][2] = p.z;
    coord_radius[i][3] = r[i];
  }
}

//initialize with indexed types and combined cr
CoordinateSet::CoordinateSet(const std::vector<float4>& cr, const std::vector<int>& t, unsigned maxt):
  coord_radius(cr.size(),4), type_index(cr.size()), max_type(maxt) {
  unsigned N = cr.size();
  if(N != t.size()) {
    throw std::invalid_argument("Types and coordinates are of different sizes");
  }

  //copy data
  memcpy(coord_radius.cpu().data(), &cr[0], sizeof(float4)*N);
  for(unsigned i = 0; i < N; i++) {
    type_index[i] = t[i];   //convert to float
  }
}

//initialize with indexed types (float) and combined cr
CoordinateSet::CoordinateSet(const std::vector<float4>& cr, const std::vector<float>& t, unsigned maxt):
  coord_radius(cr.size(),4), type_index(cr.size()), max_type(maxt) {
  unsigned N = cr.size();
  if(N != t.size()) {
    throw std::invalid_argument("Types and coordinates are of different sizes");
  }

  //copy data
  memcpy(type_index.cpu().data(), &t[0], sizeof(float)*N);
  memcpy(coord_radius.cpu().data(), &cr[0], sizeof(float4)*N);
}

///initialize with indexed types using grids - data is copied into coordinate set
CoordinateSet::CoordinateSet(const Grid2f& cr, const Grid1f& t, unsigned maxt):
    coord_radius(cr.dimension(0), cr.dimension(1)), type_index(t.dimension(0)), max_type(maxt) {
  coord_radius.copyFrom(cr);
  type_index.copyFrom(t);
}

CoordinateSet::CoordinateSet(const Grid2fCUDA& cr, const Grid1fCUDA& t, unsigned maxt):
    coord_radius(cr.dimension(0), cr.dimension(1)), type_index(t.dimension(0)), max_type(maxt) {
  coord_radius.copyFrom(cr);
  type_index.copyFrom(t);
}


inline size_t typ_vec_size(const std::vector<std::vector<float> >& t) {
  if(t.size() == 0) return 0;
  return t[0].size();
}


//initialize with vector types
CoordinateSet::CoordinateSet(const std::vector<float3>& c, const std::vector<std::vector<float> >& t, const std::vector<float>& r):
  coord_radius(c.size(),4), type_vector(c.size(),typ_vec_size(t)), max_type(typ_vec_size(t)) {

  unsigned N = c.size();
  if(N != t.size()) {
    throw std::invalid_argument("Types and coordinates are of different sizes");
  }
  if(N != r.size()) {
    throw std::invalid_argument("Radius and coordinates are of different sizes");
  }

  //copy data
  type_vector.tocpu();  coord_radius.tocpu();
  for(unsigned i = 0; i < N; i++) {
    float3 p = c[i];
    coord_radius[i][0] = p.x;
    coord_radius[i][1] = p.y;
    coord_radius[i][2] = p.z;
    coord_radius[i][3] = r[i];
  }
  memcpy(type_vector.cpu().data(), &t[0], sizeof(float)*N*max_type);

}

//initialize with vector types and combined cr
CoordinateSet::CoordinateSet(const std::vector<float4>& cr, const std::vector<std::vector<float> >& t):
  coord_radius(cr.size(),4), type_vector(cr.size(),typ_vec_size(t)), max_type(typ_vec_size(t)) {

  unsigned N = cr.size();
  if(N != t.size()) {
    throw std::invalid_argument("Types and coordinates are of different sizes");
  }

  //copy data
  memcpy(coord_radius.cpu().data(), &cr[0], sizeof(float4)*N);
  memcpy(type_vector.cpu().data(), &t[0], sizeof(float)*N*max_type);
}

//vector types in grids
CoordinateSet::CoordinateSet(const Grid2f& cr, const Grid2f& t):
    coord_radius(cr.dimension(0),cr.dimension(1)), type_vector(t.dimension(0), t.dimension(1)), max_type(t.dimension(1)) {
  coord_radius.copyFrom(cr);
  type_vector.copyFrom(t);
}

CoordinateSet::CoordinateSet(const Grid2fCUDA& cr, const Grid2fCUDA& t):
    coord_radius(cr.dimension(0),cr.dimension(1)), type_vector(t.dimension(0), t.dimension(1)), max_type(t.dimension(1)) {
  coord_radius.copyFrom(cr);
  type_vector.copyFrom(t);
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
  unsigned N = coord_radius.dimension(0);
  if(N == 0) return ret;

  coord_radius.tocpu(); //todo, gpuize
  for(unsigned i = 0; i < N; i++) {
    ret.x += coord_radius(i,0);
    ret.y += coord_radius(i,1);
    ret.z += coord_radius(i,2);
  }
  ret.x /= N;
  ret.y /= N;
  ret.z /= N;
  return ret;
}

void CoordinateSet::dump(std::ostream& out) const {
  unsigned N = coord_radius.dimension(0);
  coord_radius.tocpu();
  if(N == 0) return;
  for(unsigned i = 0; i < N; i++) {
    out << coord_radius(i,0) << "," << coord_radius(i,1) << "," << coord_radius(i,2);
    if(has_indexed_types()) {
      out << " " << type_index(i);
    }
    //todo vector types
    out << "\n";
  }
}

void CoordinateSet::copyInto(const CoordinateSet& s) {

  coord_radius = coord_radius.resized(s.coord_radius.dimension(0), 4);
  coord_radius.copyFrom(s.coord_radius);

  type_index = type_index.resized(s.type_index.dimension(0));
  type_index.copyFrom(s.type_index);

  type_vector = type_vector.resized(s.type_vector.dimension(0), s.type_vector.dimension(1));
  type_vector.copyFrom(s.type_vector);

  max_type = s.max_type;
  src = s.src;
}



}
