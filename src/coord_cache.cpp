/*
 * coord_cache.cpp
 *
 *  Created on: Apr 12, 2019
 *      Author: dkoes
 */

#include "libmolgrid/coord_cache.h"

#include <boost/algorithm/string.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem.hpp>
#include <openbabel/obconversion.h>
#include <cuda_runtime.h>


namespace libmolgrid {

using namespace std;
using namespace OpenBabel;

unsigned do_not_optimize_away;

//read in molcache if present
CoordCache::CoordCache(std::shared_ptr<AtomTyper> t, const ExampleProviderSettings& settings,
    const std::string& mc): typer(t), data_root(settings.data_root), molcache(mc),
        use_cache(settings.cache_structs), addh(settings.add_hydrogens), make_vector_types(settings.make_vector_types) {
  if(molcache.length() > 0) {
    static_assert(sizeof(size_t) == 8, "size_t must be 8 bytes");

    if(!boost::filesystem::exists(molcache)) {
      //try looking in dataroot
      molcache = (boost::filesystem::path(data_root) / molcache).string();
    }
    ifstream mcache(molcache.c_str());
    if(!mcache) throw invalid_argument("Could not open file: "+molcache);
    int version = 0;
    mcache.read((char*)&version,sizeof(int));
    if(version != -1) {
      throw invalid_argument(molcache+" is not a valid molcache2 file");
    }
    size_t start = 0;
    mcache.read((char*)&start, sizeof(size_t));
    mcache.seekg(start);

    //open memory map for moldata
    cache_map.open(molcache.c_str(),start);
    if(!cache_map.is_open()) throw logic_error("Could not memory map "+molcache);

    //read in name + offset
    unsigned char len = 0;
    char molname[256];
    size_t offset = 0;

    while(mcache.read((char*)&len, 1)) {
      //read name then offset
      mcache.read(molname, len);
      molname[len] = 0;
      mcache.read((char*)&offset, sizeof(size_t));

      string mname(molname);
      offsets[string_cache.get(mname)] = offset;
    }

    //prefetch into file cache
    unsigned sum = 0;
    for(unsigned i = 0, n = cache_map.size(); i < n; i += 1024) {
      sum += cache_map.data()[i];
    }
    do_not_optimize_away = sum;
  }

}

//set coords using the cache
void CoordCache::set_coords(const char *fname, CoordinateSet& coord) {

  struct info {
    float x,y,z;
    int type;
  };


  if(offsets.count(fname)) {
    size_t off = offsets[fname];
    const char *data = cache_map.data()+off;
    unsigned natoms = *(unsigned*)data;
    info *atoms = (info*)(data+sizeof(unsigned));

    if(typer->is_vector_typer())
      throw invalid_argument("Vector typer used with molcache files");

    vector<float3> c; c.reserve(natoms);
    vector<float> r; r.reserve(natoms);
    vector<int> t; t.reserve(natoms);
    for(unsigned i = 0; i < natoms; i++)
    {
      info& atom = atoms[i];
      auto t_r = typer->get_int_type(atom.type);
      if(t_r.first >= 0) { //ignore neg
        t.push_back(t_r.first);
        r.push_back(t_r.second);
        c.push_back(make_float3(atom.x,atom.y,atom.z));
      }
    }

    coord = CoordinateSet(c, t, r, typer->num_types());
    coord.src = fname;
  }
  else if(memcache.count(fname)) {
    coord = memcache[fname].clone(); //always copy out of cache
  } else {
    std::string fullname = fname;
    if(data_root.length()) {
      boost::filesystem::path p = boost::filesystem::path(data_root) / boost::filesystem::path(fname);
      fullname = p.string();
    }
    //check for custom gninatypes file
    if(boost::algorithm::ends_with(fname,".gninatypes"))
    {
      if(typer->is_vector_typer())
        throw invalid_argument("Vector typer used with gninatypes files");

      ifstream in(fullname.c_str());
      if(!in) throw invalid_argument("Could not read "+fullname);

      vector<float3> c;
      vector<float> r;
      vector<int> t;
      info atom;

      while(in.read((char*)&atom, sizeof(atom)))
      {
        auto t_r = typer->get_int_type(atom.type);
        if(t_r.first >= 0) { //ignore neg
          t.push_back(t_r.first);
          r.push_back(t_r.second);
          c.push_back(make_float3(atom.x,atom.y,atom.z));
        }
      }

      coord = CoordinateSet(c, t, r, typer->num_types());
      coord.src = fname;
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

      coord = CoordinateSet(&mol, *typer);
      coord.src = fname;
    } else {
      coord = CoordinateSet();
      coord.max_type = typer->num_types(); //empty, but include type size
    }

    AtomIndexTyper *ityper = dynamic_cast<AtomIndexTyper*>(typer.get());
    if(make_vector_types && ityper) {
      coord.make_vector_types(false, ityper->get_type_radii());
    }
    if(use_cache) { //save coord
      memcache[fname] = coord.clone(); //save a copy in case the returned set is modified
    }
  }
}


} /* namespace libmolgrid */
