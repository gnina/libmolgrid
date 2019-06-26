/*
 * transform.cpp
 *
 *  Created on: Jan 10, 2019
 *      Author: dkoes
 */


#include "libmolgrid/transform.h"

namespace libmolgrid {

Transform::Transform(float3 c, float random_translate /*= 0.0*/, bool random_rotate /*= false*/): center(c) {
      std::uniform_real_distribution<double> R(-1.0,1);
      translate.x = R(random_engine)*random_translate;
      translate.y = R(random_engine)*random_translate;
      translate.z = R(random_engine)*random_translate;

      if(random_rotate) {
        std::uniform_real_distribution<double> unit_sample(0,1);
        double u1 = unit_sample(random_engine);
        double u2 = unit_sample(random_engine);
        double u3 = unit_sample(random_engine);
        double sq1 = sqrt(1-u1);
        double sqr = sqrt(u1);
        double r1 = sq1*sin(2*M_PI*u2);
        double r2 = sq1*cos(2*M_PI*u2);
        double r3 = sqr*sin(2*M_PI*u3);
        double r4 = sqr*cos(2*M_PI*u3);

        Q = Quaternion(r1,r2,r3,r4);
      } //else Quaternion constructor is identity
}

void Transform::forward(const Example& in, Example& out, bool dotranslate) const {
  //transform each coordset
  if(in.sets.size() != out.sets.size()) {
    throw std::invalid_argument("Incompatible example sizes"); //todo, resize out
  }
  for(unsigned i = 0, n = in.sets.size(); i < n; i++) {
    forward(in.sets[i],out.sets[i],dotranslate);
  }
}

void Transform::forward(const CoordinateSet& in, CoordinateSet& out, bool dotranslate) const {
  if(in.coords.dimension(0) != out.coords.dimension(0)) {
    throw std::invalid_argument("Incompatible coordinateset sizes"); //todo, resize out
  }
  if(in.coords.ongpu()) {
    forward(in.coords.gpu(), out.coords.gpu(), dotranslate);
  } else {
    forward(in.coords.cpu(), out.coords.cpu(), dotranslate);
  }

}

template <typename Dtype>
void Transform::forward(const Grid<Dtype, 2, false>& in, Grid<Dtype, 2, false>& out, bool dotranslate /*=true*/) const {
  checkGrids(in,out);
  unsigned N = in.dimension(0);

  for(unsigned i = 0; i < N; i++) {
    Dtype x,y,z;
    x = in(i,0);
    y = in(i,1);
    z = in(i,2);

    float3 newpt = make_float3(0,0,0);
    if(dotranslate) {
      newpt = Q.transform(x,y,z,center, translate);
    } else {
      newpt = Q.transform(x,y,z,center, newpt);
    }

    out[i][0] = newpt.x;
    out[i][1] = newpt.y;
    out[i][2] = newpt.z;
  }
}

template void Transform::forward(const Grid<float, 2, false>& in, Grid<float, 2, false>&, bool) const;
template void Transform::forward(const Grid<double, 2, false>& in, Grid<double, 2, false>&, bool) const;

//inverse, translate, then unrotate
template <typename Dtype>
void Transform::backward(const Grid<Dtype, 2, false>& in, Grid<Dtype, 2, false>& out, bool dotranslate /*=true*/) const {
  checkGrids(in,out);
  unsigned N = in.dimension(0);
  Quaternion invQ = Q.inverse();

  float3 untranslate;
  if(dotranslate)
    untranslate = make_float3(-translate.x - center.x, -translate.y - center.y, -translate.z - center.z);
  else
    untranslate = make_float3(-center.x, -center.y, -center.z);

  for(unsigned i = 0; i < N; i++) {
    Dtype x,y,z;
    x = in[i][0] + untranslate.x;
    y = in[i][1] + untranslate.y;
    z = in[i][2] + untranslate.z;

    float3 newpt = invQ.rotate(x,y,z);
    out[i][0] = newpt.x + center.x;
    out[i][1] = newpt.y + center.y;
    out[i][2] = newpt.z + center.z;
  }
}

template void Transform::backward(const Grid<float, 2, false>&, Grid<float, 2, false>&, bool) const;
template void Transform::backward(const Grid<double, 2, false>&, Grid<double, 2, false>&, bool) const;

};

