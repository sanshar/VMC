#include<Complex.h>

//This is a bit ugly because we are having to make a special case of the function with double
Complex<stan::math::var> operator*(const Complex<stan::math::var> &LHS, const double &RHS)
{
  Complex<stan::math::var> zcopy(LHS);
  zcopy.real() *= RHS; 
  zcopy.imag() *= RHS; 
  return zcopy;
}

Complex<stan::math::var> operator*(const double &RHS, const Complex<stan::math::var> &LHS)
{
  Complex<stan::math::var> zcopy(LHS);
  zcopy.real() *= RHS; 
  zcopy.imag() *= RHS; 
  return zcopy;
}

Complex<stan::math::var> operator/(const double &LHS, const Complex<stan::math::var> &RHS)
{
  Complex<stan::math::var> zcopy(LHS);
  return zcopy /= RHS;
}


Complex<stan::math::var> operator/(const Complex<stan::math::var> &LHS, const double &RHS)
{
  Complex<stan::math::var> zcopy(LHS);
  return zcopy /= RHS;
}

