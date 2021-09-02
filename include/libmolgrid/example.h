/** \file example.h
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
#include "libmolgrid/coordinateset.h"

namespace libmolgrid {

enum IterationScheme { Continuous = 0, LargeEpoch = 1, SmallEpoch = 2};

#define MAKE_SETTINGS() \
    EXSET(bool, shuffle, false, "randomize order of examples") \
    EXSET(bool, balanced, false, "provide equal number of positive and negative examples as determined by label") \
    EXSET(bool, stratify_receptor, false, "sample uniformly across receptors (first molecule)") \
    EXSET(int, labelpos, 0, "position of binary label") \
    EXSET(int, stratify_pos, 1, "position of label for numerical stratification") \
    EXSET(bool, stratify_abs, true, "stratify based on absolute value, for cases where negative has special meaning (e.g., hinge loss indicator)") \
    EXSET(float, stratify_min, 0, "minimum range for value stratification") \
    EXSET(float, stratify_max, 0, "maximum range for value stratification") \
    EXSET(float, stratify_step, 0, "step size for value stratification, together with min and max determines number of bins") \
    EXSET(int, group_batch_size, 1, "slice time series (groups) by batches of this size") \
    EXSET(int, max_group_size, 0, "maximum group size, all groups are padded out to this size; example file must contain group number in first column") \
    EXSET(size_t, default_batch_size, 1, "default batch size") \
    EXSET(bool, cache_structs, true, "retain coordinates in memory for faster training") \
    EXSET(bool, add_hydrogens, true, "protonate read in molecule using openbabel") \
    EXSET(bool, duplicate_first, false, "clone the first coordinate set to be paired with each of the remaining (receptor-ligand pairs)") \
    EXSET(size_t, num_copies, 1, "number of times to repeatedly produce an example") \
    EXSET(bool, make_vector_types, false, "convert index types into one-hot encoded vector types") \
    EXSET(IterationScheme, iteration_scheme, Continuous, "how to iterate over examples; note that the last batch may get padded with example from the next epoch ") \
    EXSET(std::string, data_root, "", "prefix for data files") \
    EXSET(std::string, recmolcache, "", "precalculated molcache2 file for receptor (first molecule); if doesn't exist, will look in data _root") \
    EXSET(std::string, ligmolcache, "", "precalculated molcache2 file for ligand; if doesn't exist, will look in data_root")

/** Description of how examples should be provided
 * This provides configuration to example refs, extractors, and the provider itself
 * as a declarative syntax.
 */
struct ExampleProviderSettings {
    #define EXSET(TYPE, NAME, DEFAULT, DOC) TYPE NAME = DEFAULT;
    MAKE_SETTINGS()
};


// Docstring_Example
/** \brief A single example represented by its typed coordinates and label(s)
 *
 */
struct Example {

    //indexed  by atom group
    std::vector<CoordinateSet> sets;
    std::vector<float> labels;
    int group = -1;
    bool seqcont = false; ///for grouped inputs, true if not first member of group

    /// The total number of atom across all sets
    size_t num_coordinates() const;

    /// The maximum number of types across all sets - if unique_index_types is true, each set gets different type ids
    size_t num_types(bool unique_index_types=true) const;

    /// Accumulate sum of each type class into sum
    template<bool isCUDA>
    void sum_types(Grid<float, 1, isCUDA>& sum, bool unique_types=true) const;

    // Docstring_Example_merge_coordinates_1
    /** \brief Combine all coordinate sets into one and return it.
     * All coordinate sets must have the same kind of typing.  The result is a copy of the input coordinates.
     * @param[in] start ignore coordinates sets prior to this index (default zero)
     * @param[in] unique_indexed_types if true, different coordinate sets will have unique, non-overlapping types
     *
     */
    CoordinateSet merge_coordinates(unsigned start = 0, bool unique_index_types=true) const;

    // Docstring_Example_merge_coordinates_2
    /** \brief Combine all coordinate sets into one.
     * All coordinate sets must have index typing
     * @param[out] coords  combined coordinates
     * @param[out] type_index combined types
     * @param[out] radii combined radii
     * @param[in] start ignore coordinates sets prior to this index (default zero)
     * @param[in] unique_indexed_types if true, different coordinate sets will have unique, non-overlapping types
     */
    void merge_coordinates(Grid2f& coords, Grid1f& type_index, Grid1f& radii, unsigned start=0, bool unique_index_types=true) const;
    void merge_coordinates(std::vector<float3>& coords, std::vector<float>& type_index, std::vector<float>& radii, unsigned start=0, bool unique_index_types=true) const;

    // Docstring_Example_merge_coordinates_3
    /** \brief Combine all coordinate sets into one.
     * All coordinate sets must have vector typing
     * @param[out] coords  combined coordinates
     * @param[out] type_index combined types
     * @param[out] radii combined radii
     * @param[in] start ignore coordinates sets prior to this index (default zero)
     * @param[in] unique_indexed_types if true, different coordinate sets will have unique, non-overlapping types
     */
    void merge_coordinates(Grid2f& coords, Grid2f& type_vector, Grid1f& radii, unsigned start=0, bool unique_index_types=true) const;
    void merge_coordinates(std::vector<float3>& coords, std::vector<std::vector<float> >& type_vector, std::vector<float>& radii, unsigned start=0, bool unique_index_types=true) const;

    // Docstring_Example_extract_labels
    /** \brief Extract labels from a vector of examples, as returned by ExampleProvider.next_batch.
     *
     * @param[in] examples  vector of examples
     * @param[out] grid 2D grid (NxL)
     */
    template <bool isCUDA>
    static void extract_labels(const std::vector<Example>& examples, Grid<float, 2, isCUDA>& out);

    // Docstring_Example_extract_label
    /** \brief Extract a specific label from a vector of examples, as returned by ExampleProvider.next_batch.
     *
     * @param[in] examples  vector of examples
     * @param[in] labelpos position of label
     * @param[out] out 2D grid (NxL)
     */
    template <bool isCUDA>
    static void extract_label(const std::vector<Example>& examples, unsigned labelpos, Grid<float, 1, isCUDA>& out);
    
    //pointer equality, implemented for python vector
    bool operator==(const Example& rhs) const {
      return sets == rhs.sets && labels == rhs.labels;
    }

    void togpu() { for(unsigned i = 0, n = sets.size(); i < n; i++) { sets[i].togpu(); } }
    void tocpu() { for(unsigned i = 0, n = sets.size(); i < n; i++) { sets[i].tocpu(); } }

    /// Convert coordinate sets to vector types
    void make_vector_types() { for(unsigned i = 0, n = sets.size(); i < n; i++) { sets[i].make_vector_types(); } }

    /// Return true if all coord_sets >= start have vector types (or is empty)
    bool has_vector_types(unsigned start = 0) const;

    /// Return true if all coord_sets >= start have index types (or is empty)
    bool has_index_types(unsigned start = 0) const;
};

/** \brief a reference to a single example - the parsed line.  This is distinct from an
 * Example to enable out-of-core training (although the default should be to load all examples
 * into memory).
 */
struct ExampleRef {
    std::vector<const char*> files;
    std::vector<float> labels;
    int group = -1;
    bool seqcont = false; ///true if not first frame of group

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

} /* namespace libmolgrid */

#endif /* EXAMPLE_H_ */
