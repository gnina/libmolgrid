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
#include "example.h"

namespace libmolgrid {

using namespace std;

StringCache string_cache;

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
