#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include <boost/algorithm/string.hpp>
#include "gaussianBasis.h"

using namespace std;
using namespace Eigen;

void gaussianBasis::read()
{
  string basisFile = "basisInfo.txt";
  ifstream basisin(basisFile.c_str());
  shls_slice.resize(2);
  

    basisin >> IntegralType;
    basisin >> norbs;
    basisin >> shls_slice[0];
    basisin >> shls_slice[1];
    int n,m;
    
    basisin >> n;
    aoloc.resize(n, 0);
    for (int i=0; i<n; i++)
      basisin >> aoloc[i];
    
    basisin >> n >> m;
    atm.resize(n*m);
    natm = n;
    for (int i=0; i<n; i++)
      for (int j=0; j<m; j++)
        basisin >> atm[i*m+j];
    
    basisin >> n >> m;
    bas.resize(n*m);
    nbas = n;
    for (int i=0; i<n; i++)
      for (int j=0; j<m; j++)
        basisin >> bas[i*m+j];
    
    basisin >> n;
    env.resize(n, 0.);
    for (int i=0; i<n; i++)
      basisin >> env[i];
    
    basisin >> n;
    n = 10000;
    non0table.resize(n, 0.);
    for (int i=0; i<n; i++)
      non0table[i] = 1;

}

int gaussianBasis::getNorbs() {
  return norbs;
}
  
void gaussianBasis::eval(const vector<Vector3d>& x, vector<double>& values) {
  int ngrids = x.size();

  vector<Vector3d>& xcopy = const_cast<vector<Vector3d>&>(x);
  
  if (boost::iequals(IntegralType, "sph")) {
    GTOval_sph(ngrids, &shls_slice[0], &aoloc[0],
                &values[0], &xcopy[0][0], &non0table[0], &atm[0],
                natm, &bas[0], nbas, &env[0]);
  }
  else
    GTOval_cart(ngrids, &shls_slice[0], &aoloc[0],
               &values[0], &xcopy[0][0], &non0table[0], &atm[0],
               natm, &bas[0], nbas, &env[0]);

}

void gaussianBasis::eval(const Vector3d& x, double* values) {
  int ngrids = 1;
  
  //vector<Vector3d>& xcopy = const_cast<vector<Vector3d>&> x;
  Vector3d& xcopy = const_cast<Vector3d&>(x);
  if (boost::iequals(IntegralType, "sph")) {
    GTOval_sph(ngrids, &shls_slice[0], &aoloc[0],
                &values[0], &xcopy[0], &non0table[0], &atm[0],
                natm, &bas[0], nbas, &env[0]);
  }
  else
    GTOval_cart(ngrids, &shls_slice[0], &aoloc[0],
                &values[0], &xcopy[0], &non0table[0], &atm[0],
                natm, &bas[0], nbas, &env[0]);

}

void gaussianBasis::eval_deriv2(const Vector3d& x, double* values) {
  int ngrids = 1;
  
  Vector3d& xcopy = const_cast<Vector3d&>(x);
  if (boost::iequals(IntegralType, "sph")) {
    GTOval_sph_deriv2(ngrids, &shls_slice[0], &aoloc[0],
                      &values[0], &xcopy[0], &non0table[0], &atm[0],
                      natm, &bas[0], nbas, &env[0]);
  }
  else
    GTOval_cart_deriv2(ngrids, &shls_slice[0], &aoloc[0],
                       &values[0], &xcopy[0], &non0table[0], &atm[0],
                       natm, &bas[0], nbas, &env[0]);

}

