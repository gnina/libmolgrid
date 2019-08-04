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

#include "libmolgrid/example.h"

namespace libmolgrid {

using namespace std;

StringCache string_cache;

size_t Example::coordinate_size() const {
  unsigned N = 0;
  for(unsigned i = 0, n = sets.size(); i < n; i++) {
    N += sets[i].coords.dimension(0);
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
void Example::merge_coordinates(Grid2f& c, Grid1f& t, Grid1f& r, unsigned start, bool unique_index_types) const {

  vector<float3> coords;
  vector<float> types;
  vector<float> radii;

  merge_coordinates(coords, types, radii, start, unique_index_types);

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

void Example::merge_coordinates(std::vector<float3>& coords, std::vector<float>& types, std::vector<float>& radii, unsigned start, bool unique_index_types) const {
  unsigned N = coordinate_size();

  coords.clear();
  types.clear();
  radii.clear();

  if(sets.size() <= start) return;

  coords.reserve(N);
  types.reserve(N);
  radii.reserve(N);

  //accumulate info
  unsigned toffset = 0; //amount to offset types
  for(unsigned s = start, ns = sets.size(); s < ns; s++) {
    const CoordinateSet& CS = sets[s];
    unsigned n = CS.coords.dimension(0);
    if(n == 0) continue; //ignore empties

    if(!CS.has_indexed_types()) throw logic_error("Coordinate sets do not have compatible index types for merge.");

    //todo: memcpy this
    for(unsigned i = 0; i < n; i++) {
      auto cr = CS.coords[i];
      coords.push_back(make_float3(cr[0],cr[1],cr[2]));
      types.push_back(CS.type_index[i]+toffset);
      radii.push_back(CS.radii[i]);
    }
    if(unique_index_types) toffset += CS.max_type;
  }
}

//grid version of vector
void Example::merge_coordinates(Grid2f& c, Grid2f& t, Grid1f& r, unsigned start, bool unique_index_types) const {

  vector<float3> coords;
  vector< vector<float> > types;
  vector<float> radii;

  merge_coordinates(coords, types, radii, start, unique_index_types);

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

void Example::merge_coordinates(std::vector<float3>& coords, std::vector<std::vector<float> >& types,
    std::vector<float>& radii, unsigned start, bool unique_index_types) const {

  coords.clear();
  types.clear();
  radii.clear();

  if(sets.size() <= start) return;

  unsigned N = coordinate_size();
  unsigned maxt = sets[start].max_type;
  //validate type vector sizes
  for(unsigned i = start, n = sets.size(); i < n; i++) {
    if(sets[i].coords.dimension(0) == 0)
      continue;
    if(!sets[i].has_vector_types())
      throw logic_error("Coordinate sets do not have compatible vector types for merge.");

    if(maxt == 0) //skip over empty sets
      maxt = sets[i].max_type;
    if(sets[i].type_vector.dimension(1) != maxt)
      throw logic_error("Coordinate sets do not have compatible sized vector types. "+itoa(sets[i].type_vector.dimension(1)) + " vs " + itoa(maxt));
  }

  coords.reserve(N);
  types.reserve(N);
  radii.reserve(N);

  //accumulate info
  for(unsigned s = start, ns = sets.size(); s < ns; s++) {
    const CoordinateSet& CS = sets[s];
    unsigned n = CS.coords.dimension(0);
    if(n == 0) continue;

    //todo: memcpy this
    for(unsigned i = 0; i < n; i++) {
      auto cr = CS.coords[i];
      coords.push_back(make_float3(cr[0],cr[1],cr[2]));

      types.push_back(vector<float>(maxt));
      vector<float>& tvec = types.back();
      memcpy(&tvec[0], CS.type_vector[i].cpu().data(), sizeof(float)*maxt);
    }
    for(unsigned i = 0, nr = CS.radii.size(); i < nr; i++) {
      radii.push_back(CS.radii[i]);
    }
  }
}

CoordinateSet Example::merge_coordinates(unsigned start, bool unique_index_types) const {

  //if all sets are vector types, merge as such - empty sets are ambiguous so have to check all
  bool has_vector_types = true;
  for(unsigned i = start, n = sets.size(); i < n; i++) {
    if(!sets[i].has_vector_types())
      has_vector_types = false;
  }
  if(sets.size() <= start) {
    return CoordinateSet();
  } else if(sets.size() == start+1) {
    //copy data for consistency with multiple sets
    return sets[start].clone();
  } else if(!has_vector_types) {

    vector<float3> coords;
    vector<float> types;
    vector<float> radii;
    merge_coordinates(coords, types, radii, start, unique_index_types);

    return CoordinateSet(coords, types, radii, type_size(unique_index_types));

  } else { //vector types

    vector<float3> coords;
    vector<vector<float> > types;
    vector<float> radii;
    merge_coordinates(coords, types, radii, start, unique_index_types);
    return CoordinateSet(coords, types, radii);
  }
}

template <bool isCUDA>
void Example::extract_labels(const vector<Example>& examples, Grid<float, 2, isCUDA>& out) {
  if(out.dimension(0) != examples.size()) throw std::out_of_range("Grid dimension does not match number of examples: "+itoa(out.dimension(0)) + " vs "+itoa(examples.size()));
  if(examples.size() == 0) return;
  size_t nlabels = examples[0].labels.size();
  if(nlabels != out.dimension(1)) throw std::out_of_range("Grid dimension does not match number of labels: "+itoa(nlabels)+ " vs "+itoa(out.dimension(1)));

  for(unsigned i = 0, n = examples.size(); i < n; i++) {
    const vector<float>& labels = examples[i].labels;
    if(labels.size() != nlabels) throw logic_error("Non-uniform number of labels: "+itoa(nlabels) +" vs "+ itoa(labels.size()));
    if(isCUDA) {
      LMG_CUDA_CHECK(cudaMemcpy(out[i].data(), &labels[0], sizeof(float)*nlabels, cudaMemcpyHostToDevice));
    } else {
      memcpy(out[i].data(), &labels[0], sizeof(float)*nlabels);
    }
  }
}

template void Example::extract_labels(const vector<Example>& examples, Grid<float, 2, false>& out);
template void Example::extract_labels(const vector<Example>& examples, Grid<float, 2, true>& out);


template <bool isCUDA>
void Example::extract_label(const std::vector<Example>& examples, unsigned labelpos, Grid<float, 1, isCUDA>& out) {
  unsigned N = examples.size();
  if(out.dimension(0) != N) throw std::out_of_range("Grid dimension does not match number of examples");
  if(N == 0) return;
  size_t nlabels = examples[0].labels.size();
  if(labelpos >= nlabels) throw std::out_of_range("labelpos invalid: " +itoa(labelpos) + " >= " + itoa(nlabels));

  //unpack into a single vector
  vector<float> labels(N);
  for(unsigned i = 0; i < N; i++) {
    if(labelpos >= labels.size()) throw std::out_of_range("labelpos invalid (nonuniform labels): " +itoa(labelpos) + " >= " + itoa(labels.size()));
    labels[i] = examples[i].labels[labelpos];
  }
   if(isCUDA) {
     LMG_CUDA_CHECK(cudaMemcpy(out.data(), &labels[0], sizeof(float)*N, cudaMemcpyHostToDevice));
   } else {
     memcpy(out.data(), &labels[0], sizeof(float)*N);
   }
}

template void Example::extract_label(const vector<Example>&, unsigned, Grid<float, 1, false>& );
template void Example::extract_label(const vector<Example>&, unsigned, Grid<float, 1, true>& );


bool is_numeric(const string& number)
{
    char* end = nullptr;
    strtod(number.c_str(), &end);  //if entire string is number, end will be set to end of string

    return end != nullptr && *end == 0;  //could also check end != number.c_str() if whole string doesn't need to be numerical
}

ExampleRef::ExampleRef(const std::string& line, int numlabels, bool hasgroup) {
  stringstream stream(line);
  string tmp;

  if(numlabels < 0) { //auto detect
    vector<string> tokens;
    boost::algorithm::split(tokens, line,boost::is_any_of("\t "),boost::token_compress_on);
    for(unsigned i = 0, n = tokens.size(); i < n; i++) {
      numlabels = i;
      if(!is_numeric(tokens[i]))
        break;
    }
    if(hasgroup) numlabels--;
  }

  //get group if needed
  if(hasgroup) {
    stream >> group;
  }

  //grab all labels
  double label = 0;
  labels.reserve(numlabels);
  for(int i = 0; i < numlabels; i++) {
    stream >> label;
    labels.push_back(label);
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
