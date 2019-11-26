#pragma once
#include <Eigen/Dense>
#include <vector>
using namespace std;
using namespace Eigen;

template <
  typename DerivedX,
  typename DerivedR,
  typename DerivedC,
  typename DerivedY>
void Slice(
    const Eigen::DenseBase<DerivedX> & X,
    const vector<DerivedR> & R,
    const vector<DerivedC> & C,
    DerivedY & Y)
{
  int ym = Y.rows();
  int yn = Y.cols();

  // loop over output rows, then columns
  for(int i = 0;i<ym;i++)
  {
    for(int j = 0;j<yn;j++)
    {
      Y(i,j) = X(R[i],C[j]);
    }
  }
};

class calcRDM{
  Matrix2cd rdmval2;
  Matrix3cd rdmval3;
  Matrix4cd rdmval4;
  MatrixXcd rdmval5;
  MatrixXcd rdmval6;

  vector<int> rows, cols;
 public:
  calcRDM() {
    rows.resize(6,0); cols.resize(6,0);
    rdmval5.resize(5,5);
    rdmval6.resize(6,6);
  }
  
  complex<double> getRDM(int a, int b, MatrixXcd& rdm);
  complex<double> getRDM(int a, int b, int c, int d, MatrixXcd& rdm);
  complex<double> getRDM(int a, int b, int c, int d, int e, int f, MatrixXcd& rdm);
  complex<double> getRDM(int a, int b, int c, int d, int e, int f, int g, int h, MatrixXcd& rdm);
  complex<double> getRDM(int a, int b, int c, int d, int e, int f, int g, int h, int i, int j, MatrixXcd& rdm);
  complex<double> getRDM(int a, int b, int c, int d, int e, int f, int g, int h, int i, int j, int k, int l, MatrixXcd& rdm);

  complex<double> calcTerm1(int P, int Q, int I, int K, MatrixXcd& rdm);
  complex<double> calcTerm1(int P, int Q, int I, int J, int K, int L, MatrixXcd& rdm);
  complex<double> calcTerm2(int P, int Q, int I, int K, int R, int S, MatrixXcd& rdm);
  complex<double> calcTerm2(int P, int Q, int I, int J, int K, int L, int R, int S, MatrixXcd& rdm);
  
};
