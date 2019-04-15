/*
 * test_quaternion.cpp
 *
 *  Created on: Jan 7, 2019
 *      Author: dkoes
 */

#define BOOST_TEST_MODULE quaternion_test
#include <boost/test/unit_test.hpp>

#include "libmolgrid/quaternion.h"
#include <boost/math/quaternion.hpp>

using namespace libmolgrid;

typedef boost::math::quaternion<float> boostQ;

template<class Q1, class Q2>
static void eqQ(Q1 a, Q2 b)
{
  //are these the same?
  BOOST_CHECK_EQUAL(a.R_component_1(), b.R_component_1());
  BOOST_CHECK_EQUAL(a.R_component_2(), b.R_component_2());
  BOOST_CHECK_EQUAL(a.R_component_3(), b.R_component_3());
  BOOST_CHECK_EQUAL(a.R_component_4(), b.R_component_4());

  BOOST_CHECK_EQUAL(a.real(), b.real());
}

BOOST_AUTO_TEST_CASE( quaternion_operatons )
{
  // should get same results as boost quaternion
  boostQ bq1(.2,.3,.5,.7);
  boostQ bq2(.5,-0.5,-0.5,-0.5);

  Quaternion q1(.2,.3,.5,.7);
  Quaternion q2(.5,-0.5,-0.5,-0.5);

  eqQ(q1, bq1);
  eqQ(q2, bq2);

  boostQ bqtmp = bq1*bq2;
  Quaternion qtmp = q1*q2;

  eqQ(qtmp, bqtmp);
  eqQ(q1, bq1);
  eqQ(q2, bq2);

  bqtmp = bq1/bq2;
  qtmp = q1/q2;

  eqQ(qtmp, bqtmp);
  eqQ(q1, bq1);
  eqQ(q2, bq2);

  BOOST_CHECK_EQUAL(q1.norm(), boost::math::norm(bq1));
  BOOST_CHECK_NE(0, q1.norm());

  qtmp = q1*q2*q2.inverse();
  BOOST_CHECK_EQUAL(qtmp.real(), q1.real()); //some floating poitn error in the rest

  //rotating
  float3 pt = make_float3(5,0,0);
  float3 center = make_float3(0,1,0);
  float3 translate = make_float3(-1,2,.5);

  float3 newpt = q1.rotate(pt.x,pt.y,pt.z);

  boostQ bpt(0,pt.x,pt.y,pt.z);
  bpt = bq1*bpt*(boost::math::conj(bq1)/boost::math::norm(bq1));

  BOOST_CHECK_EQUAL(newpt.x, bpt.R_component_2());
  BOOST_CHECK_EQUAL(newpt.y, bpt.R_component_3());
  BOOST_CHECK_EQUAL(newpt.z, bpt.R_component_4());


  float3 newpt2 = q1.transform(3,2,1, center, translate);

  bpt = boostQ(0,3-center.x,2-center.y,1-center.z);
  bpt = bq1*bpt*(boost::math::conj(bq1)/boost::math::norm(bq1));
  float3 bpt2 = make_float3(bpt.R_component_2()+center.x+translate.x,
      bpt.R_component_3()+center.y+translate.y,
      bpt.R_component_4()+center.z+translate.z);

  BOOST_CHECK_EQUAL(newpt2.x, bpt2.x);
  BOOST_CHECK_EQUAL(newpt2.y, bpt2.y);
  BOOST_CHECK_EQUAL(newpt2.z, bpt2.z);

}
