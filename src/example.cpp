/*
 * Example.cpp
 *
 *  Created on: Feb 27, 2019
 *      Author: dkoes
 */
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <unordered_set>
#include <boost/algorithm/string.hpp>
#include <cuda_runtime.h>

#include "example.h"

namespace libmolgrid {

using namespace std;

StringCache string_cache;

size_t Example::coordinate_size() const {
  unsigned N = 0;
  for(unsigned i = 0, n = sets.size(); i < n; i++) {
    N += sets[i].coord.dimension(0);
  }
  return N;
}

size_t Example::type_size(bool unique_index_types) const {
  unsigned maxt = 0;
  for(unsigned i = 0, n = sets.size(); i < n; i++) {
    if(unique_index_types)
      maxt += sets[i].max_type;
    else
      maxt = max(maxt, sets[i].max_type);
  }
  return maxt;
}

//grid version
void Example::merge_coordinates(Grid2f& c, Grid1f& t, Grid1f& r, bool unique_index_types) const {

  vector<float3> coords;
  vector<float> types;
  vector<float> radii;

  merge_coordinates(coords, types, radii, unique_index_types);

  //validate sizes
  if(c.dimension(0) != coords.size()) {
    throw invalid_argument("Coordinates do not have correct dimension: "+itoa(c.dimension(0)) + " != " + itoa(coords.size()));
  }
  if(c.dimension(1) != 3) {
    throw invalid_argument("Coordinates do not have correct second dimension (3): "+itoa(c.dimension(1)));
  }
  if(t.size() != types.size()) {
    throw invalid_argument("Types do not have correct dimension: "+itoa(t.size())+ " != " +itoa(types.size()));
  }
  if(r.size() != radii.size()) {
    throw invalid_argument("Radii do not have correct dimension: "+itoa(r.size())+ " != " +itoa(radii.size()));
  }

  //copy data
  memcpy(c.data(), &coords[0], sizeof(float3)*coords.size());
  memcpy(t.data(), &types[0], sizeof(float)*types.size());
  memcpy(r.data(), &radii[0], sizeof(float)*radii.size());

}

void Example::merge_coordinates(std::vector<float3>& coords, std::vector<float>& types, std::vector<float>& radii, bool unique_index_types) const {
  unsigned N = coordinate_size();

  coords.clear();
  types.clear();
  radii.clear();

  if(sets.size() == 0) return;

  coords.reserve(N);
  types.reserve(N);
  radii.reserve(N);

  //accumulate info
  unsigned toffset = 0; //amount to offset types
  for(unsigned s = 0, ns = sets.size(); s < ns; s++) {
    const CoordinateSet& CS = sets[s];
    unsigned n = CS.radius.size();

    if(!CS.has_indexed_types()) throw logic_error("Coordinate sets do not have compatible index types for merge.");

    //todo: memcpy this
    for(unsigned i = 0; i < n; i++) {
      coords.push_back(make_float3(CS.coord[i][0], CS.coord[i][1], CS.coord[i][2]));
      types.push_back(CS.type_index[i]+toffset);
      radii.push_back(CS.radius[i]);
    }
    if(unique_index_types) toffset += CS.max_type;
  }
}

//grid version of vector
void Example::merge_coordinates(Grid2f& c, Grid2f& t, Grid1f& r, bool unique_index_types) const {

  vector<float3> coords;
  vector< vector<float> > types;
  vector<float> radii;

  merge_coordinates(coords, types, radii, unique_index_types);

  if(types.size() == 0)
    return;

  //validate sizes
  if(c.dimension(0) != coords.size()) {
    throw invalid_argument("Coordinates do not have correct dimension: "+itoa(c.dimension(0)) + " != " + itoa(coords.size()));
  }
  if(c.dimension(1) != 3) {
    throw invalid_argument("Coordinates do not have correct second dimension (3): "+itoa(c.dimension(1)));
  }
  if(t.dimension(0) != types.size()) {
    throw invalid_argument("Types do not have correct dimension: "+itoa(t.dimension(0))+ " != " +itoa(types.size()));
  }
  if(t.dimension(1) != types[0].size()) {
    throw invalid_argument("Types do not have correct dimension: "+itoa(t.dimension(1))+ " != " +itoa(types[0].size()));
  }
  if(r.size() != radii.size()) {
    throw invalid_argument("Radii do not have correct dimension: "+itoa(r.size())+ " != " +itoa(radii.size()));
  }

  //copy data
  memcpy(c.data(), &coords[0], sizeof(float3)*coords.size());
  memcpy(t.data(), &types[0], sizeof(float)*types.size());
  memcpy(r.data(), &radii[0], sizeof(float)*radii.size());

}

void Example::merge_coordinates(std::vector<float3>& coords, std::vector<std::vector<float> >& types, std::vector<float>& radii, bool unique_index_types) const {

  coords.clear();
  types.clear();
  radii.clear();

  if(sets.size() == 0) return;

  unsigned N = coordinate_size();
  unsigned maxt = sets[0].max_type;
  //validate type vector sizes
  for(unsigned i = 0, n = sets.size(); i < n; i++) {
    if(!sets[i].has_vector_types())
      throw logic_error("Coordinate sets do not have compatible vector types for merge.");

    if(sets[i].type_vector.dimension(1) != maxt)
      throw logic_error("Coordinate sets do not have compatible sized vector types.");
  }

  coords.reserve(N);
  types.reserve(N);
  radii.reserve(N);

  //accumulate info
  for(unsigned s = 0, ns = sets.size(); s < ns; s++) {
    const CoordinateSet& CS = sets[s];
    unsigned n = CS.radius.size();
    //todo: memcpy this
    for(unsigned i = 0; i < n; i++) {
      coords.push_back(make_float3(CS.coord[i][0], CS.coord[i][1], CS.coord[i][2]));

      types.push_back(vector<float>(maxt));
      vector<float>& tvec = types.back();
      memcpy(&tvec[0], CS.type_vector[i].data(), sizeof(float)*maxt);

      radii.push_back(CS.radius[i]);
    }
  }
}

CoordinateSet Example::merge_coordinates(bool unique_index_types) const {
  if(sets.size() == 0) {
    return CoordinateSet();
  } else if(sets.size() == 1) {
    //the super easy case, possibly dangerous since not copying data?
    return sets[0];
  } else if(sets[0].has_indexed_types()) {

    vector<float3> coords;
    vector<float> types;
    vector<float> radii;
    merge_coordinates(coords, types, radii, unique_index_types);

    return CoordinateSet(coords, types, radii, type_size(unique_index_types));

  } else { //vector types

    vector<float3> coords;
    vector<vector<float> > types;
    vector<float> radii;
    merge_coordinates(coords, types, radii, unique_index_types);
    return CoordinateSet(coords, types, radii);
  }
}


bool is_numeric(const string& number)
{
    char* end = nullptr;
    strtod(number.c_str(), &end);  //if entire string is number, end will be set to end of string

    return end != nullptr && *end == 0;  //could also check end != number.c_str() if whole string doesn't need to be numerical
}

ExampleRef::ExampleRef(const std::string& line, int numlabels, bool hasgroup) {
  stringstream stream(line);
  string tmp;

  if(numlabels < 0) { //auto detect, assume no group
    vector<string> tokens;
    boost::algorithm::split(tokens, line,boost::is_any_of("\t "),boost::token_compress_on);
    for(unsigned i = 0, n = tokens.size(); i < n; i++) {
      numlabels = i;
      if(!is_numeric(tokens[i]))
        break;
    }
    if(hasgroup) numlabels--;
  }
  //grab all labels
  double label = 0;
  labels.reserve(numlabels);
  for(int i = 0; i < numlabels; i++) {
    stream >> label;
    labels.push_back(label);
  }

  //get group if needed
  if(hasgroup) {
    stream >> group;
  }

  //remainder of the line should be whitespace spearated file names
  vector<const char *> names;
  while(stream) {
    tmp.clear();
    stream >> tmp;
    if(tmp.length() == 0 || tmp[0] == '#') //nothing left or hit comment character
      break;
    const char* n = string_cache.get(tmp);
    names.push_back(n);
  }

  if(names.size() == 0) {
    throw std::invalid_argument("Missing molecular data in line: "+line);
  }
  files.reserve(names.size());
  files = names;

}



} /* namespace libmolgrid */
