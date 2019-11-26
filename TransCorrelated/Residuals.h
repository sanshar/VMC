#pragma once

#include <Eigen/Dense>
#include <vector>
#include "Complex.h"
#include "Determinants.h"
using namespace std;

using namespace Eigen;
using DiagonalXd = Eigen::DiagonalMatrix<double, Eigen::Dynamic>;

int index(int I, int J) ;
int ABAB(const int &orb);

template<typename F>
void loopOver1epar(F& fun){

  int norbs = Determinant::norbs;

  int niter = 2*norbs * 2*norbs;
  for (int orb12 = commrank; orb12 <  niter ; orb12 += commsize) {
    int orb1 = orb12/(2*norbs), orb2 = orb12%(2*norbs);
    double integral = I1( ABAB(orb1), ABAB(orb2));      
    if (abs(integral) < schd.epsilon) continue;
    
    fun(orb1, orb2, integral);
  }
};

template<typename F>
void loopOver1e(F& fun){

  int norbs = Determinant::norbs;
  
  for (int orb1 = 0; orb1 <  2 * norbs ; orb1++) {
    for (int orb2 = 0; orb2 < 2*norbs; orb2++) {
      double integral = I1( ABAB(orb1), ABAB(orb2));      
      if (abs(integral) < schd.epsilon) continue;

      fun(orb1, orb2, integral);
    }
  }
};

template<typename F>
void loopOver2epar(F& fun){

  int norbs = Determinant::norbs;

  long norb12 = 2*norbs * (2*norbs-1)/2;
  long norb1234 = norb12*norb12;
  for (size_t i = commrank; i < norb1234; i+= commsize) {
    size_t orb12 = i/norb12, orb34 = i%norb12;
    size_t orb1 = floor((-1 + pow(1 + 4*2*orb12, 0.5))/2)+1;
    size_t orb2 = orb12 - orb1*(orb1-1)/2;
    size_t orb3 = floor((-1 + pow(1 + 4*2*orb34, 0.5))/2)+1;
    size_t orb4 = orb34 - orb3*(orb3-1)/2;
    
    double integral = (I2(ABAB(orb1), ABAB(orb4), ABAB(orb2), ABAB(orb3))
                       - I2(ABAB(orb2), ABAB(orb4), ABAB(orb1), ABAB(orb3)));
    
    if (abs(integral) < schd.epsilon) continue;
    fun(orb1, orb2, orb3, orb4, integral);
  }
};

template<typename F>
void loopOver2e(F& fun){

  int norbs = Determinant::norbs;
  
  for (int orb1 = 0; orb1 < 2*norbs; orb1++) {
    for (int orb2 = 0; orb2 < orb1; orb2++) {
      for (int orb3 = 0; orb3 < 2*norbs; orb3++)
        for (int orb4 = 0; orb4 < orb3; orb4++) {
          double integral = (I2(ABAB(orb1), ABAB(orb4), ABAB(orb2), ABAB(orb3))
                             - I2(ABAB(orb2), ABAB(orb4), ABAB(orb1), ABAB(orb3)));
            
          if (abs(integral) < schd.epsilon) continue;
          fun(orb1, orb2, orb3, orb4, integral);
        }
    }
  }
};

template<typename F>
void loopOverLowerTriangle(F& fun) {
  
  int norbs = Determinant::norbs;
  for (int orbn = 0; orbn < 2*norbs; orbn++) 
    for (int orbm = 0; orbm <= orbn; orbm++) 
      fun(orbn, orbm);
};


/*
template<bool calcE          = true,
         bool calcJRes       = true,
         bool calcOrbRes     = true,
         bool calcJJHess     = false,
         bool calcOrbJHess   = false,
         bool calcOrbOrbHess = false>
struct TranscorrelatedFunctions {
  double E;
  VectorXd JRes;
  VectorXd OrbRes;

  void operator()(MatrixXcd& bra,
                  MatrixXcd& ket,
                  vector<complex<double>>& coeff,
                  VectorXd& Jastrow) {
    size_t norbs = Determinant::norbs;
    VectorXd diagcre(2*norbs), diagdes(2*norbs);

    E = 0.0;
    
    for (size_t o1o2 = 0 ; o1o2 < 16*norbs*norbs; o1o2++) {
      getCreDesDiagMatrix(diagcre, diagdes, o1, o2, norbs, Jastrow);
      MatrixXcd LambdaD = diagdes.asDiagonal()*ket,
          LambdaC = diagcre.asDiagonal()*bra;
      MatrixXcd S = LambdaC.adjoint() * LambdaD;
      MatrixXcd Sinv = S.inverse(); complex<double> Sdet = S.determinant();
      MatrixXcd rdm = (LambdaD * Sinv ) * LambdaC.adjoint();
      complex<double> factor = integral * Sdet * coeff;
      
      if (calcE)
        E += rdm(o2, o1) * integral * Sdet * coeff;
      if (calcJRes)
        
    }
  }
}
*/
struct GetResidual {
  MatrixXcd& orbitals;
  int ngrid;

  GetResidual(MatrixXcd& pOrbitals,
              int pngrid=4) : orbitals(pOrbitals), ngrid(pngrid) {};

  //get both orbital and jastrow residues
  double getResidue(const VectorXd& variables,
                    VectorXd& residual,
                    MatrixXd& JastrowHessian,
                    bool getJastrowResidue = true,
                    bool getOrbitalResidue = true,
                    bool getJastrowHessian = false);

  double getResidueSingleKet(
      double detovlp,
      MatrixXcd& bra,
      vector<MatrixXcd>& ket,
      vector<complex<double>>& coeff,
      MatrixXcd& braResidue,
      VectorXd& Jastrow,
      VectorXd& JastrowResidue,
      MatrixXd& JastrowHessian,
      bool getJastrowResidue = true,
      bool getOrbitalResidue = true,
      bool getJastrowHessian = false) ;

  //get jastrow variables
  double getJastrowResidue(const VectorXd& jastrowVars,
                           const VectorXd& braVars,
                           VectorXd& jastrowResidue);
  
  //get jastrow variables
  double getOrbitalResidue(const VectorXd& jastrowVars,
                           const VectorXd& braVars,
                           VectorXd& jastrowResidue);
  
  //get jastrow variables
  double getJastrowLM(const VectorXd& jastrowVars,
                      const VectorXd& braVars,
                      MatrixXd& Hmat,
                      VectorXd& pvec);
};

void fillWfnfromJastrow(VectorXd& JA, MatrixXd& Jtmp) ;
MatrixXcd fillWfnOrbs(MatrixXcd& orbitals, VectorXd& variables) ;

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

void fillJastrowfromWfn(MatrixXd& Jtmp, VectorXd& JA);



