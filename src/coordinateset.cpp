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
  vector<float> rads; rads.reserve(mol->NumAtoms());
  vector<float> vec;

  FOR_ATOMS_OF_MOL(a, mol){
    OBAtom *atom = &*a; //convert from iterator

    if(typer.is_vector_typer()) {

      float radius = typer.get_atom_type_vector(atom, vec);
      if(radius > 0) { //don't ignore
        c.push_back(make_float3(atom->GetX(), atom->GetY(), atom->GetZ()));
        vector_types.push_back(vec);
        rads.push_back(radius);
      }
    } else {
      auto type_rad = typer.get_atom_type_index(atom);
      int type = type_rad.first;
      float r = type_rad.second;
      if(type >= (int)max_type) throw invalid_argument("Invalid type");
      if(type >= 0) { //don't ignore atom
        c.push_back(make_float3(atom->GetX(), atom->GetY(), atom->GetZ()));
        types.push_back(type);
        rads.push_back(r);
      }
    }
  }

  //allocate grids and initialize
  unsigned N = c.size();
  coords = MGrid2f(N,3);
  assert(sizeof(float3)*N == sizeof(float)*coords.size());
  memcpy(coords.cpu().data(), &c[0], sizeof(float3)*N);

  radii = MGrid1f(N);
  memcpy(radii.cpu().data(), &rads[0], sizeof(float)*N);

  if(typer.is_vector_typer()) {
    type_vector = MGrid2f(N,max_type);
    for(unsigned i = 0; i < N; i++) {
        memcpy(type_vector[i].cpu().data(), &vector_types[i][0], sizeof(float)*max_type);
    }
  } else {
    type_index = MGrid1f(N);
    memcpy(type_index.cpu().data(), &types[0], sizeof(float)*N);
  }
}

//initialize with indexed types
CoordinateSet::CoordinateSet(const std::vector<float3>& c, const std::vector<int>& t, const std::vector<float>& r, unsigned maxt):
  coords(c.size(),3), type_index(c.size()), radii(c.size()), max_type(maxt) {
  unsigned N = c.size();
  if(N != t.size()) {
    throw std::invalid_argument("Types and coordinates are of different sizes");
  }
  if(N != r.size()) {
    throw std::invalid_argument("Radii and coordinates are of different sizes");
  }

  //copy data
  type_index.tocpu(); coords.tocpu(); radii.tocpu();
  memcpy(radii.cpu().data(), &r[0], sizeof(float)*N);
  assert(sizeof(float3)*N == sizeof(float)*coords.size());
  memcpy(coords.cpu().data(), &c[0], sizeof(float3)*N);

  for(unsigned i = 0; i < N; i++) {
    type_index[i] = t[i];   //convert to float
  }
}

//initialize with indexed types (float)
CoordinateSet::CoordinateSet(const std::vector<float3>& c, const std::vector<float>& t, const std::vector<float>& r, unsigned maxt):
  coords(c.size(),3), type_index(c.size()), radii(c.size()), max_type(maxt) {
  unsigned N = c.size();
  if(N != t.size()) {
    throw std::invalid_argument("Types and coordinates are of different sizes");
  }
  if(N != r.size()) {
    throw std::invalid_argument("Radii and coordinates are of different sizes");
  }

  //copy data
  type_index.tocpu(); coords.tocpu(); radii.tocpu();
  memcpy(type_index.cpu().data(), &t[0], sizeof(float)*N);
  memcpy(radii.cpu().data(), &r[0], sizeof(float)*N);
  assert(sizeof(float3)*N == sizeof(float)*coords.size());
  memcpy(coords.cpu().data(), &c[0], sizeof(float3)*N);
}

///initialize with indexed types using grids - data is copied into coordinate set
CoordinateSet::CoordinateSet(const Grid2f& c, const Grid1f& t, const Grid1f& r, unsigned maxt):
    coords(c.dimension(0), c.dimension(1)), type_index(t.dimension(0)), radii(r.dimension(0)), max_type(maxt) {
  coords.copyFrom(c);
  type_index.copyFrom(t);
  radii.copyFrom(r);
}

CoordinateSet::CoordinateSet(const Grid2fCUDA& c, const Grid1fCUDA& t, const Grid1fCUDA& r, unsigned maxt):
    coords(c.dimension(0), c.dimension(1)), type_index(t.dimension(0)), radii(r.dimension(0)), max_type(maxt) {
  coords.copyFrom(c);
  type_index.copyFrom(t);
  radii.copyFrom(r);
}

inline size_t typ_vec_size(const std::vector<std::vector<float> >& t) {
  if(t.size() == 0) return 0;
  return t[0].size();
}


//initialize with vector types
CoordinateSet::CoordinateSet(const std::vector<float3>& c, const std::vector<std::vector<float> >& t, const std::vector<float>& r):
  coords(c.size(),3), type_vector(c.size(),typ_vec_size(t)), radii(r.size()), max_type(typ_vec_size(t)) {

  unsigned N = c.size();
  if(N != t.size()) {
    throw std::invalid_argument("Types and coordinates are of different sizes");
  }
  if(N != r.size() && max_type != r.size()) {
    throw std::invalid_argument("Radius and coordinates/types are of different sizes: "+itoa(N)+" "+itoa(r.size())+" "+itoa(max_type));
  }

  //copy data
  type_vector.tocpu();  coords.tocpu(); radii.tocpu();
  memcpy(radii.cpu().data(), &r[0], sizeof(float)*r.size());
  assert(sizeof(float3)*N == sizeof(float)*coords.size());
  memcpy(coords.cpu().data(), &c[0], sizeof(float3)*N);

  for(unsigned i = 0; i < N; i++) {
    memcpy(type_vector[i].cpu().data(), &t[i][0], sizeof(float)*t[i].size());
  }

}

//vector types in grids
CoordinateSet::CoordinateSet(const Grid2f& c, const Grid2f& t, const Grid1f& r):
    coords(c.dimension(0),c.dimension(1)), type_vector(t.dimension(0), t.dimension(1)), radii(r.dimension(0)), max_type(t.dimension(1)) {
  coords.copyFrom(c);
  type_vector.copyFrom(t);
  radii.copyFrom(r);
}

CoordinateSet::CoordinateSet(const Grid2fCUDA& c, const Grid2fCUDA& t, const Grid1fCUDA& r):
    coords(c.dimension(0),c.dimension(1)), type_vector(t.dimension(0), t.dimension(1)), radii(r.dimension(0)), max_type(t.dimension(1)) {
  coords.copyFrom(c);
  type_vector.copyFrom(t);
  radii.copyFrom(r);
}


CoordinateSet::CoordinateSet(const CoordinateSet& rec, const CoordinateSet& lig, bool unique_index_types):
  coords(rec.coords.dimension(0)+lig.coords.dimension(0), 3),
  type_index(rec.type_index.dimension(0)+lig.type_index.dimension(0)),
  type_vector(rec.type_vector.dimension(0)+lig.type_vector.dimension(0), rec.type_vector.dimension(1)),
  radii(rec.radii.dimension(0)+lig.radii.dimension(0)) {

  mergeInto(rec, lig, unique_index_types);
}


///convert index types to vector types in-place
void CoordinateSet::make_vector_types(bool include_dummy_type, const std::vector<float>& type_radii) {
  unsigned N = type_index.size();

  if(type_radii.size() > 0 && type_radii.size() != max_type) {
    throw invalid_argument("Type radii size " + itoa(type_radii.size()) + " does not equal max type "+itoa(max_type));
  }

  if(include_dummy_type)
    max_type++; //add a type that doesn't match any atom

  type_vector = MGrid2f(N, max_type); //grid are always zero initialized
  for(unsigned i = 0; i < N; i++) {
    unsigned t = type_index[i];
    if(t < max_type) {
      type_vector(i,t) = 1.0;
    }
  }

  if(type_radii.size()>  0) {
    //change radii from being indexed by atom to being indexed by type
    radii = radii.resized(max_type);
    radii.tocpu();
    if(include_dummy_type) radii[max_type-1] = 0.0;
    memcpy(radii.data(), &type_radii[0], sizeof(float)*type_radii.size());
  }
}

float3 CoordinateSet::center() const {
  float3 ret = make_float3(0,0,0);
  unsigned N = coords.dimension(0);
  if(N == 0) return ret;

  coords.tocpu(); //todo, gpuize
  for(unsigned i = 0; i < N; i++) {
    ret.x += coords(i,0);
    ret.y += coords(i,1);
    ret.z += coords(i,2);
  }
  ret.x /= N;
  ret.y /= N;
  ret.z /= N;
  return ret;
}

void CoordinateSet::dump(std::ostream& out) const {
  unsigned N = coords.dimension(0);
  coords.tocpu();
  if(N == 0) return;
  for(unsigned i = 0; i < N; i++) {
    out << coords(i,0) << "," << coords(i,1) << "," << coords(i,2);
    if(has_indexed_types()) {
      out << " " << type_index(i);
    }
    //todo vector types
    out << "\n";
  }
}

void CoordinateSet::size_like(const CoordinateSet& s) {
  coords = coords.resized(s.coords.dimension(0), 3);
  type_index = type_index.resized(s.type_index.dimension(0));
  type_vector = type_vector.resized(s.type_vector.dimension(0), s.type_vector.dimension(1));
  radii = radii.resized(s.radii.dimension(0));
}

void CoordinateSet::copyInto(const CoordinateSet& s) {
  size_like(s);
  coords.copyFrom(s.coords);
  type_index.copyFrom(s.type_index);
  type_vector.copyFrom(s.type_vector);
  radii.copyFrom(s.radii);

  max_type = s.max_type;
  src = s.src;
}

void CoordinateSet::mergeInto(const CoordinateSet& rec, const CoordinateSet& lig, bool unique_index_types) {

  coords = coords.resized(rec.coords.dimension(0)+lig.coords.dimension(0), 3);
  type_index = type_index.resized(rec.type_index.dimension(0)+lig.type_index.dimension(0));
  type_vector = type_vector.resized(rec.type_vector.dimension(0)+lig.type_vector.dimension(0), rec.type_vector.dimension(1));
  radii = radii.resized(rec.radii.dimension(0)+lig.radii.dimension(0));

  unsigned NR = rec.coords.dimension(0);
  unsigned NL = lig.coords.dimension(0);
  unsigned num_rec_types = rec.max_type;

  if(rec.type_vector.dimension(1) != lig.type_vector.dimension(1)) {
    throw std::invalid_argument("Type vectors are incompatible sizes");
  }
  if(rec.has_vector_types() != lig.has_vector_types() || rec.has_indexed_types() != lig.has_indexed_types()) {
    throw std::invalid_argument("Incompatible types when combining coodinate sets");
  }
  if(rec.has_indexed_types()) {
    if(unique_index_types)
      max_type = rec.max_type + lig.max_type;
    else
      max_type = max(rec.max_type,lig.max_type);
  } else {
    if(rec.max_type != lig.max_type)
      throw std::invalid_argument("Type vectors are incompatible sizes, weirdly"); //should be checked above
  }

  coords.copyFrom(rec.coords);
  type_index.copyFrom(rec.type_index);
  type_vector.copyFrom(rec.type_vector);
  radii.copyFrom(rec.radii);

  coords.copyInto(NR, lig.coords);
  type_index.copyInto(NR, lig.type_index);
  type_vector.copyInto(NR, lig.type_vector);
  radii.copyInto(NR, lig.radii);

  if(unique_index_types) {
    //todo: gpuize
    bool isgpu = type_index.ongpu();
    type_index.tocpu();
    for(unsigned i = NR; i < NR+NL; i++) {
      type_index[i] += num_rec_types;
    }
    if(isgpu) type_index.togpu();
  }
}

//copy w/index types
template<bool isCUDA>
size_t CoordinateSet::copyTo(Grid<float, 2, isCUDA>& c, Grid<float, 1, isCUDA>& t, Grid<float, 1, isCUDA>& r) const {
  if(c.dimension(1) != 3) throw invalid_argument("Coordinates have wrong secondary dimension in copyTo (3 != "+itoa(coords.dimension(1)));
  size_t ret = coords.copyTo(c);
  type_index.copyTo(t);
  radii.copyTo(r);
  return ret / 3;
}

template size_t CoordinateSet::copyTo(Grid<float, 2, false>& c, Grid<float, 1, false>& t,
    Grid<float, 1, false>& r) const;
template size_t CoordinateSet::copyTo(Grid<float, 2, true>& c, Grid<float, 1, true>& t, Grid<float, 1, true>& r) const;

//copy w/vector types
template<bool isCUDA>
size_t CoordinateSet::copyTo(Grid<float, 2, isCUDA>& c, Grid<float, 2, isCUDA>& t, Grid<float, 1, isCUDA>& r) const {
  if(coords.dimension(1) != 3) throw invalid_argument("Coordinates have wrong secondary dimension in copyTo (3 != "+itoa(coords.dimension(1)));
  size_t ret = coords.copyTo(c);
  radii.copyTo(r);

  if(t.dimension(1) != type_vector.dimension(1)) {
    //copy a row at a time
    size_t rows = min(t.dimension(0), type_vector.dimension(0));
    for(unsigned i = 0; i < rows; i++) {
      Grid<float, 1, isCUDA> dst(t[i]);
      type_vector[i].copyTo(dst);
    }

  } else { //straight copy is fine
    type_vector.copyTo(t);
  }

  return ret / 3;
}

template size_t CoordinateSet::copyTo(Grid<float, 2, false>& c, Grid<float, 2, false>& t, Grid<float, 1, false>& r) const;
template size_t CoordinateSet::copyTo(Grid<float, 2, true>& c, Grid<float, 2, true>& t, Grid<float, 1, true>& r) const;


void CoordinateSet::sum_types(Grid<float, 1, false>& sum, bool zerofirst) const {
  if(sum.dimension(0) != num_types())
    throw invalid_argument("Type sum output dimension does not match number of types: "+itoa(sum.dimension(0))+" vs "+itoa(num_types()));

  if(zerofirst) sum.fill_zero();
  int NT = num_types();
  if(!has_vector_types()) {
    for(unsigned i = 0, n = type_index.dimension(0); i < n; i++) {
      int t = round(type_index[i]);
      if(t < 0) continue;
      if(t >= NT) throw out_of_range("Somehow an index type is too large (internal error).");
      sum[t] += 1.0;
    }
  } else { //vector types
    for(unsigned i = 0, n = type_vector.dimension(0); i < n; i++) {
      //i is atom index
      for(unsigned j = 0, m = type_vector.dimension(1); j < m; j++) {
        //j is type index
        sum[j] += type_vector[i][j];
      }
    }
  }
}


}
