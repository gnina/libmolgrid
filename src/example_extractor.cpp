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
  ex.group = ref.group;
  ex.seqcont = ref.seqcont;

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
    ret += coord_caches[t].num_types();
  }
  if(duplicate_poses && coord_caches.size() > 2) {
    size_t rsize = coord_caches[0].num_types();
    size_t dups = coord_caches.size() - 2;
    ret += rsize*dups;
  }
  return ret;
}
size_t ExampleExtractor::num_types() const {
  return count_types(coord_caches.size());
}

size_t ExampleExtractor::num_types(const ExampleRef& ref) const {
  return count_types(ref.files.size());
}

std::vector<std::string> ExampleExtractor::get_type_names() const {
  vector<string> ret;
  for(unsigned i = 0, n = coord_caches.size(); i < n; i++) {
    vector<string> names = coord_caches[i].get_type_names();
    for(auto name : names) {
      ret.push_back(itoa(i)+"_"+name);
    }
  }
  return ret;
}

} /* namespace libmolgrid */
