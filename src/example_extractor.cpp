/*
 * example_extractor.cpp
 *
 *  Created on: Mar 25, 2019
 *      Author: dkoes
 */

#include "libmolgrid/example_extractor.h"
#include "libmolgrid/atom_typer.h"

#include <boost/algorithm/string.hpp>
#include <boost/filesystem/path.hpp>
#include <openbabel/obconversion.h>
#include <cuda_runtime.h>

namespace libmolgrid {

using namespace std;
using namespace OpenBabel;

void ExampleExtractor::extract(const ExampleRef& ref, Example& ex) {
  ex.labels = ref.labels; //the easy part

  //for each file in ref, get a coordinate set using the matching typer
  ex.sets.clear();

  if(!duplicate_poses || ref.files.size() < 3) {
    ex.sets.resize(ref.files.size());
    for(unsigned i = 0, n = ref.files.size(); i < n; i++) {
      const char* fname = ref.files[i];
      unsigned t = i;
      if(t >= coord_caches.size()) t = coord_caches.size()-1; //repeat last typer if necessary
      coord_caches[t].set_coords(fname, ex.sets[i]);
    }
  } else { //duplicate first pose (receptor) to match each of the remaining poses
    unsigned N = ref.files.size() - 1;
    ex.sets.resize(N*2);
    coord_caches[0].set_coords(ref.files[0], ex.sets[0]);
    for(unsigned i = 1, n = ref.files.size(); i < n; i++) {
      const char* fname = ref.files[i];
      unsigned t = i;
      if(t >= coord_caches.size()) t = coord_caches.size()-1; //repeat last typer if necessary
      coord_caches[t].set_coords(fname, ex.sets[2*(i-1)+1]);

      //duplicate receptor by copying
      if(i > 1) ex.sets[2*(i-1)] = ex.sets[0];
    }
  }

}

//assume there are n files, return number oftypes
size_t ExampleExtractor::count_types(unsigned n) const {
  size_t ret = 0;
  for(unsigned i = 0; i < n; i++) {
    unsigned t = i;
    if(t >= coord_caches.size()) t = coord_caches.size()-1;
    ret += coord_caches[t].type_size();
  }
  if(duplicate_poses && coord_caches.size() > 2) {
    size_t rsize = coord_caches[0].type_size();
    size_t dups = coord_caches.size() - 2;
    ret += rsize*dups;
  }
  return ret;
}
size_t ExampleExtractor::type_size() const {
  return count_types(coord_caches.size());
}

size_t ExampleExtractor::type_size(const ExampleRef& ref) const {
  return count_types(ref.files.size());
}


} /* namespace libmolgrid */
