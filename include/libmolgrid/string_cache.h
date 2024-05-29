/** \file string_cache.h
 *
 *  Class for avoiding duplicate memory allocation of identical strings.
 *  Use const char*
 */

#ifndef STRING_CACHE_H_
#define STRING_CACHE_H_

#include <string>
#include <unordered_set>

namespace libmolgrid {

//for memory efficiency, only store a given string once and use the const char*
class StringCache {
  std::unordered_set<std::string> strings;
public:
  const char* get(const std::string& s)
  {
    strings.insert(s);
    //we assume even as the set is resized that strings never get allocated
    return strings.find(s)->c_str();
  }
};

extern StringCache string_cache;

}; //namespace libmolgrid
#endif // STRING_CACHE_H_