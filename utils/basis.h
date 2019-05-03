#ifndef BASIS_HEADER_H
#define BASIS_HEADER_H

#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include <boost/serialization/serialization.hpp>
using namespace std;
using namespace Eigen;


struct Basis {
 private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
  }
  
 public:
  
  virtual void read() = 0;
  virtual int getNorbs() = 0;
  virtual void maxCoord(vector<Vector3d>& maxr) = 0;
  virtual void eval(const vector<Vector3d>& x, vector<double>& values) = 0;
  virtual void eval(const Vector3d& x, double* values) = 0;
  virtual void eval_deriv2(const Vector3d& x, double* values) = 0;

};


#endif

