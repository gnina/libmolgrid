/** \file coord_cache.h - class for caching coordinates with types
 *
 *  Created on: Apr 12, 2019
 *      Author: dkoes
 */

#ifndef COORD_CACHE_H_
#define COORD_CACHE_H_

#include "libmolgrid/atom_typer.h"
#include "libmolgrid/coordinateset.h"
#include "libmolgrid/example.h"
#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/unordered_map.hpp>
namespace libmolgrid {

/** \brief Load and cache molecular coordinates and atom types.
 *
 *  Precalculated molcache2 files are supported and are
 *  memory mapped for efficient memory usage when running multiple
 *  training runs.
 */
class CoordCache {
  using MemCache = std::unordered_map<const char *, CoordinateSet>;
  MemCache memcache;
  std::shared_ptr<AtomTyper> typer;
  std::string data_root;
  std::string molcache;
  bool use_cache = true; // is possible to disable caching
  bool addh = true;      /// protonate
  bool make_vector_types =
      false; /// convert index types to vector, will also convert to type based
             /// radii and add a dummy type

  // for memory mapped cache
  boost::iostreams::mapped_file_source cache_map;
  std::unordered_map<const char *, size_t>
      offsets; // map from names to position in cache_map

public:
  CoordCache() {}
  CoordCache(
      std::shared_ptr<AtomTyper> t,
      const ExampleProviderSettings &settings = ExampleProviderSettings(),
      const std::string &mc = "");
  ~CoordCache() {}

  /** \brief Set coord to the appropriate CoordinateSet for fname
   * @param[in] fname file name, not including root directory prefix, of
   * molecular data
   * @param[out] coord  CoordinateSet for passed molecule
   */
  void set_coords(const char *fname, CoordinateSet &coord);

  /// return the number of types (channels) each example will have
  size_t num_types() const { return typer->num_types(); }

  std::vector<std::string> get_type_names() const {
    return typer->get_type_names();
  }

  /** \brief Write out current contents of memory cache to provided output
   * stream.
   * @param[in] out output stream
   */
  void save_mem_cache(std::ostream &out) const;

  /** \brief Write out current contents of memory cache to provided file.
   * @param[in] fname file name
   */
  void save_mem_cache(const std::string &fname) const {
    std::ofstream out(fname.c_str());
    if (!out)
      throw std::invalid_argument("Could not open file " + fname);
    save_mem_cache(out);
  }

  /** \brief Read contents of input stream into memory cache.
   * @param[in] in input stream
   */
  void load_mem_cache(std::istream &in);
  /** \brief Read contents of provided file into memory cache.
   * @param[in] fname file name
   */
  void load_mem_cache(const std::string &fname) {
    std::ifstream in(fname.c_str());
    if (!in)
      throw std::invalid_argument("Could not load file " + fname);
    load_mem_cache(in);
  }

  size_t mem_cache_size() const {
    return memcache.size();
  }
};

} /* namespace libmolgrid */

#endif /* COORD_CACHE_H_ */
