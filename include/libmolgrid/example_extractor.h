/**\file example_extractor.h
 *
 * Conversion of ExampleRef to Example
 *
 *
 *  Created on: Mar 25, 2019
 *      Author: dkoes
 */

#ifndef EXAMPLE_EXTRACTOR_H_
#define EXAMPLE_EXTRACTOR_H_

#include <memory>
#include <vector>
#include <unordered_map>
#include "libmolgrid/example.h"
#include "libmolgrid/exampleref_providers.h"
#include "libmolgrid/coordinateset.h"
#include "libmolgrid/coord_cache.h"

namespace libmolgrid {

/** \brief Converts an ExampleRef to and Example
 * Loads (potentially cached) data and applies atom typers to create coordinate sets.
 Takes care of in-memory caching (optional) and also supports memory mapped gnina cache files
(incurring slight overhead on recalculation of atom types in
 exchange for substantially less real mem usage).

 Can take multiple atom typers, in which case they are applied in order, with the last being repeated.

 */
class ExampleExtractor {

    std::vector<CoordCache> coord_caches; //different typers have duplicated caches
    bool duplicate_poses = false;

    size_t count_types(unsigned n) const;
  public:

    ExampleExtractor(const ExampleProviderSettings& settings, std::shared_ptr<AtomTyper> t) : duplicate_poses(settings.duplicate_first) {
      coord_caches.push_back(CoordCache(t, settings, settings.recmolcache));
    }

    ExampleExtractor(const ExampleProviderSettings& settings, std::shared_ptr<AtomTyper> t1, std::shared_ptr<AtomTyper> t2) : duplicate_poses(settings.duplicate_first) {
      coord_caches.push_back(CoordCache(t1, settings, settings.recmolcache));
      coord_caches.push_back(CoordCache(t2, settings, settings.ligmolcache));
    }

    ///setup an extract according to settings, types and molcaches
    ///if not present, will get molcaches from settings if there, repeating ligand if necessary
    ExampleExtractor(const ExampleProviderSettings& settings,
        const std::vector<std::shared_ptr<AtomTyper> >& typrs,
        std::vector<std::string> molcaches = std::vector<std::string>()) : duplicate_poses(settings.duplicate_first) {
      coord_caches.reserve(typrs.size());
      if(typrs.size() == 0) throw std::invalid_argument("Need at least one atom typer for example extractor");
      //if molcaches not provided, get them from settings
      if(molcaches.size() == 0) {
        if(settings.recmolcache.length() > 0) molcaches.push_back(settings.recmolcache);
        if(settings.ligmolcache.length() > 0) {
          while(molcaches.size() < typrs.size()) {
            molcaches.push_back(settings.ligmolcache);
          }
        }
      }

      for(unsigned i = 0, n = typrs.size(); i < n; i++) {
        if(i < molcaches.size()) {
          coord_caches.push_back(CoordCache(typrs[i], settings, molcaches[i]));
        } else {
          coord_caches.push_back(CoordCache(typrs[i], settings));
        }
      }
    }

    virtual ~ExampleExtractor() {}

    /// Extract ref into ex
    virtual void extract(const ExampleRef& ref, Example& ex);

    /// return the number of types (channels) each example will have
    /// Note: this is only accurate if types are explicitly setup.  Must provide an ExampleRef
    // if implicit typing is being used
    virtual size_t num_types() const;

    // return number types (channels) ref will produce
    virtual size_t num_types(const ExampleRef& ref) const;

    ///return names of types for explicitly typed examples
    ///type names are prepended by coordinate set index
    virtual std::vector<std::string> get_type_names() const;
};

} /* namespace libmolgrid */

#endif /* EXAMPLE_EXTRACTOR_H_ */
