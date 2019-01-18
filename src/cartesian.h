/*
 *\file cartesian.h
 *
 *  Data structures and routines for manipulating grids defined
 *  in an axis-aligned Cartesian coordinate system.
 *
 *  Includes routines for converting between grid and Cartesian
 *  coordinates and interpolating.
 */

#ifndef CARTESIAN_H_
#define CARTESIAN_H_

namespace libmolgrid {

class CartesianGridParam {
  public:
    CartesianGridParam();
    virtual ~CartesianGridParam();
};

} /* namespace libmolgrid */

#endif /* CARTESIAN_H_ */
