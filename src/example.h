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
#include <unordered_set>
#include "coordinateset.h"

namespace libmolgrid {


/** Description of how examples should be provided
 * This provides configuration to example refs, extractors, and the provider itself
 * as a declarative syntax.
 */
struct ExampleProviderSettings {
    ///randomize order of examples
    bool shuffle = false;
    /// provide equal number of positive and negative examples, labelpos must be set
    bool balanced = false;
    /// sample uniformly across receptors (first molecule)
    bool stratify_receptor = false;

    ///position of binary label for balancing
    int labelpos = 0;
    //the following are for stratifying on a numerical label
    /// position of label for numerical stratificatoin
    int stratify_pos = 1;
    ///stratify based on abs value, for cases where negative has special meaning (hinge loss indicator)
    bool stratify_abs = true;
    ///minimum range of stratificatoin
    float stratify_min = 0;
    ///maximum range of stratification
    float stratify_max = 0;
    ///step size of stratification
    float stratify_step = 0;

    //for grouped examples
    /// slice time series (groups) by batches
    int group_batch_size = 1;
    /// max group size, all groups are padded out to this size
    int max_group_size = 0;


    //extract config
    ///retain coordinates in memory for faster training
    bool cache_structs = true;
    ///protonate read in molecules with openbabel
    bool add_hydrogens = true;
    ///clone the first coordinate set to be paired with each of the remaining (think receptor ligand pairs)
    bool duplicate_first = false;

};


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
// example provider will take a bunch of settings (with reasonable defaults)
// will take desired atom typers
// create the appropriate example ref provider
// can add multiple files?
// populate it, load caches as appropriate
// provide next example interface

//need a mol2coords transformer

} /* namespace libmolgrid */

#endif /* EXAMPLE_H_ */
