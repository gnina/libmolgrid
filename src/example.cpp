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
    N += sets[i].coord_radius.dimension(0);
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
void Example::merge_coordinates(Grid2f& c, Grid1f& t, unsigned start, bool unique_index_types) const {

  vector<float4> coordrs;
  vector<float> types;

  merge_coordinates(coordrs, types, start, unique_index_types);

  //validate sizes
  if(c.dimension(0) != coordrs.size()) {
    throw invalid_argument("Coordinates do not have correct dimension: "+itoa(c.dimension(0)) + " != " + itoa(coordrs.size()));
  }
  if(c.dimension(1) != 4) {
    throw invalid_argument("Coordinates do not have correct second dimension (4): "+itoa(c.dimension(1)));
  }
  if(t.size() != types.size()) {
    throw invalid_argument("Types do not have correct dimension: "+itoa(t.size())+ " != " +itoa(types.size()));
  }

  //copy data
  memcpy(c.data(), &coordrs[0], sizeof(float4)*coordrs.size());
  memcpy(t.data(), &types[0], sizeof(float)*types.size());
}

void Example::merge_coordinates(std::vector<float4>& coordrs, std::vector<float>& types, unsigned start, bool unique_index_types) const {
  unsigned N = coordinate_size();

  coordrs.clear();
  types.clear();

  if(sets.size() <= start) return;

  coordrs.reserve(N);
  types.reserve(N);

  //accumulate info
  unsigned toffset = 0; //amount to offset types
  for(unsigned s = start, ns = sets.size(); s < ns; s++) {
    const CoordinateSet& CS = sets[s];
    unsigned n = CS.coord_radius.size();
    if(n == 0) continue; //ignore empties

    if(!CS.has_indexed_types()) throw logic_error("Coordinate sets do not have compatible index types for merge.");

    //todo: memcpy this
    for(unsigned i = 0; i < n; i++) {
      auto cr = CS.coord_radius[i];
      coordrs.push_back(make_float4(cr[0],cr[1],cr[2],cr[3]));
      types.push_back(CS.type_index[i]+toffset);
    }
    if(unique_index_types) toffset += CS.max_type;
  }
}

//grid version of vector
void Example::merge_coordinates(Grid2f& c, Grid2f& t, unsigned start, bool unique_index_types) const {

  vector<float4> coordrs;
  vector< vector<float> > types;
  vector<float> radii;

  merge_coordinates(coordrs, types, start, unique_index_types);

  if(types.size() == 0)
    return;

  //validate sizes
  if(c.dimension(0) != coordrs.size()) {
    throw invalid_argument("Coordinates do not have correct dimension: "+itoa(c.dimension(0)) + " != " + itoa(coordrs.size()));
  }
  if(c.dimension(1) != 4) {
    throw invalid_argument("Coordinates do not have correct second dimension (4): "+itoa(c.dimension(1)));
  }
  if(t.dimension(0) != types.size()) {
    throw invalid_argument("Types do not have correct dimension: "+itoa(t.dimension(0))+ " != " +itoa(types.size()));
  }
  if(t.dimension(1) != types[0].size()) {
    throw invalid_argument("Types do not have correct dimension: "+itoa(t.dimension(1))+ " != " +itoa(types[0].size()));
  }

  //copy data
  memcpy(c.data(), &coordrs[0], sizeof(float4)*coordrs.size());
  memcpy(t.data(), &types[0], sizeof(float)*types.size());
}

void Example::merge_coordinates(std::vector<float4>& coordrs, std::vector<std::vector<float> >& types, unsigned start, bool unique_index_types) const {

  coordrs.clear();
  types.clear();

  if(sets.size() <= start) return;

  unsigned N = coordinate_size();
  unsigned maxt = sets[start].max_type;
  //validate type vector sizes
  for(unsigned i = start, n = sets.size(); i < n; i++) {
    if(!sets[i].has_vector_types())
      throw logic_error("Coordinate sets do not have compatible vector types for merge.");

    if(sets[i].type_vector.dimension(1) != maxt)
      throw logic_error("Coordinate sets do not have compatible sized vector types.");
  }

  coordrs.reserve(N);
  types.reserve(N);

  //accumulate info
  for(unsigned s = start, ns = sets.size(); s < ns; s++) {
    const CoordinateSet& CS = sets[s];
    unsigned n = CS.coord_radius.size();
    if(n == 0) continue;

    //todo: memcpy this
    for(unsigned i = 0; i < n; i++) {
      auto cr = CS.coord_radius[i];
      coordrs.push_back(make_float4(cr[0],cr[1],cr[2],cr[3]));

      types.push_back(vector<float>(maxt));
      vector<float>& tvec = types.back();
      memcpy(&tvec[0], CS.type_vector[i].cpu().data(), sizeof(float)*maxt);
    }
  }
}

CoordinateSet Example::merge_coordinates(unsigned start, bool unique_index_types) const {
  if(sets.size() <= start) {
    return CoordinateSet();
  } else if(sets.size() == start+1) {
    //copy data for consistency with multiple sets
    return sets[start].clone();
  } else if(sets[start].has_indexed_types()) {

    vector<float4> coords;
    vector<float> types;
    vector<float> radii;
    merge_coordinates(coords, types, start, unique_index_types);

    return CoordinateSet(coords, types, type_size(unique_index_types));

  } else { //vector types

    vector<float4> coords;
    vector<vector<float> > types;
    merge_coordinates(coords, types, start, unique_index_types);
    return CoordinateSet(coords, types);
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
