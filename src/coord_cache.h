/** \file coord_cache.h - class for caching coordinates with types
 *
 *  Created on: Apr 12, 2019
 *      Author: dkoes
 */

#ifndef COORD_CACHE_H_
#define COORD_CACHE_H_

#include "coordinateset.h"
#include "atom_typer.h"
#include "example.h"
#include <boost/iostreams/device/mapped_file.hpp>

namespace libmolgrid {

/** \brief Load and cache molecular coordinates and atom types.
 *
 *  Precalculated molcache2 files are supported and are
 *  memory mapped for efficient memory usage when running multiple
 *  training runs.
 */
class CoordCache {
    using MemCache = std::unordered_map<const char*, CoordinateSet>;
    MemCache memcache;
    std::shared_ptr<AtomTyper> typer;
    std::string data_root;
    std::string molcache;
    bool use_cache = true; //is possible to disable caching
    bool addh = true; //protonate

    //for memory mapped cache
    boost::iostreams::mapped_file_source cache_map;
    std::unordered_map<const char*, size_t> offsets; //map from names to position in cache_map

  public:
    CoordCache() {}
    CoordCache(std::shared_ptr<AtomTyper> t, const ExampleProviderSettings& settings,
        const std::string& mc = "");
    ~CoordCache() {}

    /** \brief Set coord to the appropriate CoordinateSet for fname
     * @param[in] fname file name, not including root directory prefix, of molecular data
     * @param[out] coord  CoordinateSet for passed molecule
     */
    void set_coords(const char *fname, CoordinateSet& coord);

    /// return the number of types (channels) each example will have
    size_t type_size() const { return typer->num_types(); }
};

} /* namespace libmolgrid */

#endif /* COORD_CACHE_H_ */
