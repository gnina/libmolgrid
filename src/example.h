/** \file Example.h
 *
 *  Class for storing a single example as atomic coordinates with their associated
 *  atom types and labels.  An example consists of one or more atom sets.  Each
 *  set can be typed differently.
 *

 *  Created on: Feb 27, 2019
 *      Author: dkoes
 */

#ifndef EXAMPLE_H_
#define EXAMPLE_H_

#include <vector>
#include "coordinateset.h"

namespace libmolgrid {

/** \brief A single example represented by its typed coordinates and label(s)
 *
 */
struct Example {

    //indexed  by atom group
    std::vector<CoordinateSet> sets;
    std::vector<float> labels;
};

/** \brief a reference to a single example - the parsed line.  This is distinct from an
 * Example to enable out-of-core training (although the default should be to load all examples
 * into memory).
 */
struct ExampleRef {
    std::vector<const char*> files;
    std::vector<float> labels;
    int group = -1;

    ExampleRef() {}
    ///parse a line into an example reference - should have numlabels labels
    ExampleRef(const std::string& line, int numlabels, bool hasgroup=false);
};

// example provider will take a bunch of settings (with reasonable defaults)
// will take desired atom typers
// create the appropriate example ref provider
// can add multiple files?
// populate it, load caches as appropriate
// provide next example interface

//need a mol2coords transformer

} /* namespace libmolgrid */

#endif /* EXAMPLE_H_ */
