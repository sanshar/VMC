#include "calcRDM.h"
#include "Complex.h"
#include "stan/math.hpp"

template calcRDM<complex<double>>;
template calcRDM<Complex<stan::math::var>>;
