/** \file Example.h
 *
 *  Class for storing a single example as atomic coordinates with their associated
 *  atom types and labels.  An example consists of one or more atom groups.  Each
 *  group can be typed differently.  Each atom group may contain zero or more coordinate
 *  sets (all typed the same way, but not necessarily the same types) which typically
 *  correspond to time series data.
 *
 *  MergeExample is a transformation that will create a single dense grid of coordinates
 *  with the companion types grid that are suitable for grid making.
 *
 *  Created on: Feb 27, 2019
 *      Author: dkoes
 */

#ifndef EXAMPLE_H_
#define EXAMPLE_H_

#include <vector>
#include "managed_grid.h"

namespace libmolgrid {

/** \brief A collection of typed atomic coordinates
 *
 * Types may be specified either as an index or a dense vector.
 * Only one type formated will be initialized.
 *
 */
template<typename Dtype>
struct CoordinateSet {
  ManagedGrid<Dtype, 2> coords;
  ManagedGrid<Dtype, 1> type_index; //this should be integer
  ManagedGrid<Dtype, 2> type_vector;
  const unsigned max_type;  //for indexed types, non-inclusive max

  //initialize with indexed types
  CoordinateSet(const std::vector<float3>& c, const std::vector<unsigned>& t, unsigned maxt);

  //initialize with vector types
  CoordinateSet(const std::vector<float3>& c, const std::vector<std::vector<float> >& t);

  /// return true if index types are available
  bool has_indexed_types() const;

  /// return true if vector types are available
  bool has_vector_types() const;

  ///convert index types to vector types in-place
  void make_vector_types();

  unsigned num_types() const { return max_type; }

};

/** \brief CoordinateSets organized by group and time series
 *
 */
class Example {

    //indexed first by atom group and then by time series / pose number
    std::vector< std::vector<CoordinateSet> > groups;
  public:


    Example(unsigned numGroups);
    virtual ~Example();

    /// return number of atom groups
    unsigned num_groups() const;

    /// add a coordinate set to an existing group
    void add_coords_to_group(const CoordinateSet& c, unsigned groupid);

    CoordinateSet& get_coords(unsigned groupid, unsigned ts=0);
};

} /* namespace libmolgrid */

#endif /* EXAMPLE_H_ */
