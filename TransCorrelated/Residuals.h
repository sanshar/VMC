#pragma once

#include <Eigen/Dense>
#include <vector>
#include "Complex.h"
using namespace std;

using namespace Eigen;
using DiagonalXd = Eigen::DiagonalMatrix<double, Eigen::Dynamic>;

int index(int I, int J) ;

struct GetResidual {
  MatrixXcd& orbitals;
  int ngrid;

  GetResidual(MatrixXcd& pOrbitals,
              int pngrid=4) : orbitals(pOrbitals), ngrid(pngrid) {};

  //get both orbital and jastrow residues
  double getResidue(const VectorXd& variables,
                    VectorXd& residual,
                    bool getJastrowResidue = true,
                    bool getOrbitalResidue = true);

  double getResidueSingleKet(
      double detovlp,
      MatrixXcd& bra,
      MatrixXcd& ket,
      complex<double> coeff,
      MatrixXcd& braResidue,
      VectorXd& Jastrow,
      VectorXd& JastrowResidue,
      bool getJastrowResidue = true,
      bool getOrbitalResidue = true) ;

  //get jastrow variables
  double getJastrowResidue(const VectorXd& jastrowVars,
                           const VectorXd& braVars,
                           VectorXd& jastrowResidue);
  
  //get jastrow variables
  double getOrbitalResidue(const VectorXd& jastrowVars,
                           const VectorXd& braVars,
                           VectorXd& jastrowResidue);
  
};

//term is orb1^dag orb2
double getCreDesDiagMatrix(DiagonalXd& diagcre,
                           DiagonalXd& diagdes,
                           int orb1,
                           int orb2,
                           int norbs,
                           const VectorXd&JA) ;

//term is orb1^dag orb2^dag orb3 orb4
double getCreDesDiagMatrix(DiagonalXd& diagcre,
                           DiagonalXd& diagdes,
                           int orb1,
                           int orb2,
                           int orb3,
                           int orb4,
                           int norbs,
                           const VectorXd&JA);

void applyProjector(
    MatrixXcd& bra,
    vector<MatrixXcd>& ketvec,
    vector<complex<double> >& coeffs,
    int Sz,
    int ngrids);



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

//N_orbn orb1^dag orb2
complex<double> getRDMExpectation(
    MatrixXcd& rdm,
    int orbn,
    int orb1,
    int orb2,
    Matrix4cd& rdmval4,
    Matrix3cd& rdmval3,
    Matrix2cd& rdmval2,
    vector<int>& rows,
    vector<int>& cols);

//N_orbn N_orbm orb1^dag orb2
complex<double> getRDMExpectation(
    const MatrixXcd& rdm,
    int orbn,
    int orbm,
    int orb1,
    int orb2,
    Matrix4cd& rdmval4,
    Matrix3cd& rdmval3,
    Matrix2cd& rdmval2,
    vector<int>& rows,
    vector<int>& cols);

//N_orbn orb1^dag orb2^dag orb3 orb4
complex<double> getRDMExpectation(
    MatrixXcd& rdm,
    int orbn,
    int orb1,
    int orb2,
    int orb3,
    int orb4,
    Matrix4cd& rdmval4,
    Matrix3cd& rdmval3,
    Matrix2cd& rdmval2,
    vector<int>& rows,
    vector<int>& cols);

//N_orbn N_orbm orb1^dag orb2^dag orb3 orb4
complex<double> getRDMExpectation(
    MatrixXcd& rdm,
    int orbn,
    int orbm,
    int orb1,
    int orb2,
    int orb3,
    int orb4,
    Matrix4cd& rdmval4,
    Matrix3cd& rdmval3,
    Matrix2cd& rdmval2,
    vector<int>& rows,
    vector<int>& cols);

void fillJastrowfromWfn(MatrixXd& Jtmp, VectorXd& JA);



