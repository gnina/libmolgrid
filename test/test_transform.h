/*
 * test_transform.h
 *
 *  Created on: Jan 17, 2019
 *      Author: dkoes
 */

#ifndef TEST_TRANSFORM_H_
#define TEST_TRANSFORM_H_



#define TOL 0.0001f
using namespace libmolgrid;

inline void eqQ(Quaternion a, Quaternion b)
{
  //are these the same?
  BOOST_CHECK_EQUAL(a.R_component_1(), b.R_component_1());
  BOOST_CHECK_EQUAL(a.R_component_2(), b.R_component_2());
  BOOST_CHECK_EQUAL(a.R_component_3(), b.R_component_3());
  BOOST_CHECK_EQUAL(a.R_component_4(), b.R_component_4());

  BOOST_CHECK_EQUAL(a.real(), b.real());
}

inline void neqQ(Quaternion a, Quaternion b)
{
  //are these the same?
  BOOST_CHECK_NE(a.R_component_1(), b.R_component_1());
  BOOST_CHECK_NE(a.R_component_2(), b.R_component_2());
  BOOST_CHECK_NE(a.R_component_3(), b.R_component_3());
  BOOST_CHECK_NE(a.R_component_4(), b.R_component_4());

  BOOST_CHECK_NE(a.real(), b.real());
}


inline void eqPt(float3 a, float3 b)
{
  BOOST_CHECK_EQUAL(a.x,b.x);
  BOOST_CHECK_EQUAL(a.y,b.y);
  BOOST_CHECK_EQUAL(a.z,b.z);
}

inline void neqPt(float3 a, float3 b)
{
  BOOST_CHECK_NE(a.x,b.x);
  BOOST_CHECK_NE(a.y,b.y);
  BOOST_CHECK_NE(a.z,b.z);
}

inline void eqPt(const Grid1f& a, float3 b)
{
//  std::cout << "[" << a(0)<<","<<a(1)<<","<<a(2)<<"]  (" << b.x<<","<<b.y<<","<<b.z<<")\n";
  BOOST_CHECK_SMALL(a[0]-b.x,TOL);
  BOOST_CHECK_SMALL(a[1]-b.y,TOL);
  BOOST_CHECK_SMALL(a[2]-b.z,TOL);
}

inline void neqPt(const Grid1f& a, const Grid1f& b) {
  BOOST_CHECK_NE(a[0],b[0]);
  BOOST_CHECK_NE(a[1],b[1]);
  BOOST_CHECK_NE(a[2],b[2]);
}

inline void eqPt(const Grid1f& a, const Grid1f& b) {
  BOOST_CHECK_SMALL(a[0]-b[0],TOL);
  BOOST_CHECK_SMALL(a[1]-b[1],TOL);
  BOOST_CHECK_SMALL(a[2]-b[2],TOL);
}


#endif /* TEST_TRANSFORM_H_ */
