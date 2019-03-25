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
#include "example.h"
#include "exampleref_providers.h"
#include "atom_typer.h"

namespace libmolgrid {

/** \brief Converts an ExampleRef to and Example
 * Loads (potentially cached) data and applies atom typers to create coordinate sets.
 Takes care of in-memory caching (optional) and also supports memory mapped gnina cache files
(incurring slight overhead on recalculation of atom types in
 exchange for substantially less real mem usage).

 Can take multiple atom typers, in which case they are applied in order, with the last being repeated.

 */
class ExampleExtractor {
    std::vector<std::shared_ptr<AtomTyper> > typers;
  public:
    ExampleExtractor();
    virtual ~ExampleExtractor();
};

} /* namespace libmolgrid */

#endif /* EXAMPLE_EXTRACTOR_H_ */
