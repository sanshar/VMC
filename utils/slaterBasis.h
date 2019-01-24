#ifndef SLATERBasis_H
#define SLATERBasis_H

#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <map>
#include <math.h>
#include "basis.h"

using namespace Eigen;
using namespace std;


double STOradialNorm(double ex, int n);

double GTOradialNorm(double ex, int l);


struct slaterBasisOnAtom{
  int norbs;
  vector<int> NL;
  vector<double> exponents;
  vector<double> radialNorm;
  Vector3d coord;

  slaterBasisOnAtom() {};

  friend ostream& operator<<(ostream& os, slaterBasisOnAtom& sh) {
    for (int i=0; i<sh.exponents.size(); i++) {
      os <<sh.NL[2*i]<<":"<<sh.NL[2*i+1]<<"  "<< sh.exponents[i]<<"  ";
      os <<endl;
    }
    return os;
  }
  private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
    ar & norbs
        & NL
        & exponents
        & radialNorm
        & coord;
  }
  
};


struct slaterBasis : public Basis{
  int norbs;
  vector<slaterBasisOnAtom> atomicBasis;  
  map<string, Vector3d> atomList;
  
  void read();
  int getNorbs();
  void eval(const vector<Vector3d>& x, vector<double>& values);
  void eval(const Vector3d& x, double* values);
  void eval_deriv2(const Vector3d& x, double* values);
  
  friend ostream& operator<<(ostream& os, slaterBasis& s) {
    os << s.atomicBasis.size()<<endl;
    for (int i=0; i<s.atomicBasis.size(); i++)
      os<<s.atomicBasis[i]<<" ******** "<<endl;
    return os;
  }

  private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
    ar & boost::serialization::base_object<Basis>(*this);
    ar & norbs & atomicBasis & atomList;
  }
};


#endif
