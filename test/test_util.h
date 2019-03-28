#ifndef TEST_UTIL_H
#define TEST_UTIL_H
#include <random>
#include "grid.h"
#include "atom_typer.h"
#include "libmolgrid.h"

using namespace libmolgrid;
inline void make_mol(std::vector<float>& xcoords, std::vector<float>& ycoords, 
    std::vector<float>& zcoords, std::vector<float>& type_indices,
    std::vector<float>& radii, 
    size_t natoms = 0, size_t min_atoms = 1,
    size_t max_atoms = 200, float max_x = 25, float max_y = 25,
    float max_z = 25) {

  if (!natoms) {
    //if not provided, randomly generate the number of atoms
    std::uniform_int_distribution<int> natoms_dist(min_atoms, max_atoms + 1);
    natoms = natoms_dist(random_engine);
  }

  //randomly seed reasonable-ish coordinates and types
  std::uniform_real_distribution<float> xcoord_dist(-max_x,
      std::nextafter(max_x, FLT_MAX));
  std::uniform_real_distribution<float> ycoord_dist(-max_y,
      std::nextafter(max_y, FLT_MAX));
  std::uniform_real_distribution<float> zcoord_dist(-max_z,
      std::nextafter(max_z, FLT_MAX));
  std::uniform_int_distribution<int> type_dist(0, GninaIndexTyper::NumTypes - 1);
  GninaIndexTyper gtyper;

  //set up vector of atoms as well as types
  for (size_t i = 0; i < natoms; ++i) {
    xcoords.push_back(xcoord_dist(random_engine));
    ycoords.push_back(ycoord_dist(random_engine));
    zcoords.push_back(zcoord_dist(random_engine));
    int atype = type_dist(random_engine);
    type_indices.push_back((float)atype);
    auto ainfo = gtyper.get_int_type(atype);
    radii.push_back(ainfo.second);
  }
}

#endif /* TEST_UTIL_H */
