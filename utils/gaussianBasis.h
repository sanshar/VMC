#ifndef GAUSSIANBASIS_HEADER_H
#define GAUSSIANBASIS_HEADER_H

#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include <boost/serialization/serialization.hpp>
#include "basis.h"

using namespace std;
using namespace Eigen;

extern "C" {
  void GTOval_sph(int, int*, int*,
                  double*, double*, char*, int*,
                  int, int*, int, double*);
  
  void GTOval_cart(int, int*, int*,
                   double*, double*, char*, int*,
                   int, int*, int, double*);

  void GTOval_sph_deriv2(int, int*, int*,
                         double*, double*, char*, int*,
                         int, int*, int, double*);
  
  void GTOval_cart_deriv2(int, int*, int*,
                          double*, double*, char*, int*,
                          int, int*, int, double*);  
}

struct gaussianBasis : public Basis {
private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
    ar & boost::serialization::base_object<Basis>(*this);
    ar& IntegralType & norbs & shls_slice & aoloc & natm &
        atm & nbas & bas & env & non0table;
  }
  
 public:
  string IntegralType;
  int norbs;
  vector<int> shls_slice;
  vector<int> aoloc;
  int natm;
  vector<int> atm;
  int nbas;
  vector<int> bas;
  vector<double> env;
  vector<char> non0table;
  
  gaussianBasis () {};
  void read();
  int getNorbs();
  void eval(const vector<Vector3d>& x, vector<double>& values);
  void eval(const Vector3d& x, double* values);
  void eval_deriv2(const Vector3d& x, double* values);

};


#endif
