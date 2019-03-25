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
#include "example.h"

namespace libmolgrid {

using namespace std;
//for memory efficiency, only store a given string once and use the const char*
class StringCache {
  unordered_set<string> strings;
public:
  const char* get(const string& s)
  {
    strings.insert(s);
    //we assume even as the set is resized that strings never get allocated
    return strings.find(s)->c_str();
  }
};

static StringCache scache;

ExampleRef::ExampleRef(const std::string& line, int numlabels, bool hasgroup) {
  stringstream stream(line);
  string tmp;

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
    stream >> tmp;
    if(tmp.length() == 0 || tmp[0] == '#') //nothing left or hit comment character
      break;
    const char* n = scache.get(tmp);
    names.push_back(n);
  }

  if(names.size() == 0) {
    throw std::invalid_argument("Missing molecular data in line: "+line);
  }
  files.reserve(names.size());
  files = names;

}



} /* namespace libmolgrid */
