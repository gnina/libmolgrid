/*
 * example_extractor.cpp
 *
 *  Created on: Mar 25, 2019
 *      Author: dkoes
 */

#include "example_extractor.h"
#include <boost/algorithm/string.hpp>
#include <boost/filesystem/path.hpp>
#include <openbabel/obconversion.h>
#include <cuda_runtime.h>

namespace libmolgrid {

using namespace std;
using namespace OpenBabel;

//set coords using the which'th typer and cache
void ExampleExtractor::set_coords(const char *fname, unsigned which, CoordinateSet& coord) {
  if(coord_caches[which].count(fname)) {
    coord = coord_caches[which][fname];
  } else {
    std::string fullname = fname;
    if(data_root.length()) {
      boost::filesystem::path p = boost::filesystem::path(data_root) / boost::filesystem::path(fname);
      fullname = p.string();
    }
    //check for custom gninatypes file
    if(boost::algorithm::ends_with(fname,".gninatypes"))
    {
      if(typers[which]->is_vector_typer())
        throw invalid_argument("Vector typer used with gninatypes files");

      struct info {
        float x,y,z;
        int type;
      } atom;

      ifstream in(fullname.c_str());
      if(!in) throw invalid_argument("Could not read "+fullname);

      vector<float3> c;
      vector<float> r;
      vector<unsigned> t;
      while(in.read((char*)&atom, sizeof(atom)))
      {
        auto t_r = typers[which]->get_int_type(atom.type);
        t.push_back(t_r.first);
        r.push_back(t_r.second);
        c.push_back(make_float3(atom.x,atom.y,atom.z));
      }

      coord = CoordinateSet(c, t, r, typers[which]->num_types());
    }
    else if(!boost::algorithm::ends_with(fname,"none")) //reserved word
    {
      //read mol from file and set mol info (atom coords and grid positions)
      OBConversion conv;
      OBMol mol;
      if(!conv.ReadFile(&mol, fullname.c_str()))
        throw invalid_argument("Could not read " + fullname);

      if(addh) {
        mol.AddHydrogens();
      }

      coord = CoordinateSet(&mol, *typers[which]);

    }

    if(use_cache) { //save coord
      coord_caches[which][fname] = coord;
    }
  }
}


void ExampleExtractor::extract(const ExampleRef& ref, Example& ex) {
  ex.labels = ref.labels; //the easy part

  //for each file in ref, get a coordinate set using the matching typer
  ex.sets.clear();

  if(!duplicate_poses || ref.files.size() < 3) {
    ex.sets.resize(ref.files.size());
    for(unsigned i = 0, n = ref.files.size(); i < n; i++) {
      const char* fname = ref.files[i];
      unsigned t = i;
      if(t >= typers.size()) t = typers.size()-1; //repeat last typer if necessary
      set_coords(fname, t, ex.sets[i]);
    }
  } else { //duplicate first pose (receptor) to match each of the remaining poses
    unsigned N = ref.files.size() - 1;
    ex.sets.resize(N*2);
    set_coords(ref.files[0], 0, ex.sets[0]);
    for(unsigned i = 1, n = ref.files.size(); i < n; i++) {
      const char* fname = ref.files[i];
      unsigned t = i;
      if(t >= typers.size()) t = typers.size()-1; //repeat last typer if necessary
      set_coords(fname, t, ex.sets[2*(i-1)+1]);

      //duplicate receptor by copying
      if(i > 1) ex.sets[2*(i-1)] = ex.sets[0];
    }
  }

}


} /* namespace libmolgrid */
