#include "Complex.h"
#include <stan/math.hpp>

//constructor
Complex<stan::math::var> operator*(const Complex<stan::math::var> &LHS, const double &RHS){
  Complex<stan::math::var> zout(LHS.real()*RHS, LHS.imag()*RHS);
  return zout;
}

Complex<stan::math::var> operator*(const double &RHS, const Complex<stan::math::var> &LHS){
  Complex<stan::math::var> zout(LHS.real()*RHS, LHS.imag()*RHS);
  return zout;
}

//Complex<stan::math::var> operator/(const double &LHS, const Complex<stan::math::var> &RHS){
//Complex<stan::math::var> zout(LHS.real()*RHS, LHS.imag()*RHS);
//return zout;
//}

Complex<stan::math::var> operator/(const Complex<stan::math::var> &LHS, const double &RHS){
  Complex<stan::math::var> zout(LHS.real()/RHS, LHS.imag()/RHS);
  return zout;
}
