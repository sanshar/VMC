#pragma once

#include <Eigen/Dense>
#include <vector>
#include "Complex.h"

using namespace std;
using namespace Eigen;
using DiagonalXd = Eigen::DiagonalMatrix<double, Eigen::Dynamic>;

namespace ResidualGradient {

//term is orb1^dag orb2
template<typename T>
T getCreDesDiagMatrix(DiagonalMatrix<T, Dynamic>& diagcre,
                      DiagonalMatrix<T, Dynamic>& diagdes,
                      int orb1,
                      int orb2,
                      int norbs,
                      const Matrix<T, Dynamic,1>&JA) {
  for (int j=0; j<2*norbs; j++) {
    diagcre.diagonal()[j] = exp(-2.*JA(index(orb1, j)));
    diagdes.diagonal()[j] = exp( 2.*JA(index(orb2, j)));
  }

  T factor = exp( JA(index(orb1, orb1)) - JA(index(orb2, orb2)));
  return factor;

};

//term is orb1^dag orb2^dag orb3 orb4
template<typename T>
T getCreDesDiagMatrix(DiagonalMatrix<T, Dynamic>& diagcre,
                      DiagonalMatrix<T, Dynamic>& diagdes,
                      int orb1,
                      int orb2,
                      int orb3,
                      int orb4,
                      int norbs,
                      const Matrix<T, Dynamic, 1>&JA) {

  for (int j=0; j<2*norbs; j++) {
    diagcre.diagonal()[j] = exp(-2.*JA(index(orb1, j)) - 2.*JA(index(orb2, j)));
    diagdes.diagonal()[j] = exp(+2.*JA(index(orb3, j)) + 2.*JA(index(orb4, j)));
  }

  T factor = exp( 2*JA(index(orb1, orb2)) + JA(index(orb1, orb1)) + JA(index(orb2, orb2))
                  -2*JA(index(orb3, orb4)) - JA(index(orb3, orb3)) - JA(index(orb4, orb4))) ;
  return factor;

};

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

template<typename complexT>
complexT getRDMExpectation(Matrix<complexT, Dynamic, Dynamic>& rdm,
                           int orbn, int orbm, int orb1, int orb2, int orb3, int orb4,
                           Matrix<complexT, 4, 4>& rdmval4,
                           Matrix<complexT, 3, 3>& rdmval3,
                           Matrix<complexT, 2, 2>& rdmval2,
                           vector<int>& rows, vector<int>& cols) {

  complexT contribution(0.) ;
  bool calc4rdm = true;

  if (orbm == orb2) {
    rows[0] = orbn; rows[1] = orb3; rows[2] = orb4;
    cols[0] = orb1; cols[1] = orbm; cols[2] = orbn;
    ResidualGradient::Slice(rdm, rows, cols, rdmval3);
    contribution -= rdmval3.determinant();
    calc4rdm = false;
  }
  if (orbm == orb1) {
    rows[0] = orbn; rows[1] = orb3; rows[2] = orb4;
    cols[0] = orb2; cols[1] = orbm; cols[2] = orbn;
    ResidualGradient::Slice(rdm, rows, cols, rdmval3);
    contribution += rdmval3.determinant();
    calc4rdm = false;
  }
  if (orbn == orb2) {
    rows[0] = orbm; rows[1] = orb3; rows[2] = orb4;
    cols[0] = orb1; cols[1] = orbm; cols[2] = orbn;
    ResidualGradient::Slice(rdm, rows, cols, rdmval3);
    contribution += rdmval3.determinant();
    calc4rdm = false;
  }
  if (orbn == orb2 && orbm == orb1) {
    rows[0] = orb3; rows[1] = orb4;
    cols[0] = orbm; cols[1] = orbn;
    ResidualGradient::Slice(rdm, rows, cols, rdmval2);
    contribution -= rdmval2.determinant();
    calc4rdm = false;
  }
  if (orbn == orb1) {
    rows[0] = orbm; rows[1] = orb3; rows[2] = orb4;
    cols[0] = orb2; cols[1] = orbm; cols[2] = orbn;
    ResidualGradient::Slice(rdm, rows, cols, rdmval3);
    contribution -= rdmval3.determinant();
    calc4rdm = false;
  }
  if (orbn == orb1 && orbm == orb2) {
    rows[0] = orb3; rows[1] = orb4;
    cols[0] = orbm; cols[1] = orbn;
    ResidualGradient::Slice(rdm, rows, cols, rdmval2);
    contribution += rdmval2.determinant();
    calc4rdm = false;
  }

  if (calc4rdm) {
    rows[0] = orbm; rows[1] = orbn; rows[2] = orb3; rows[3] = orb4;
    cols[0] = orbm; cols[1] = orbn; cols[2] = orb2; cols[3] = orb1;
    
    ResidualGradient::Slice(rdm, rows, cols, rdmval4);
    contribution =  rdmval4.determinant();
  }
  
  return contribution;
};


//N_orbn orb1^dag orb2^dag orb3 orb4
template<typename complexT>
complexT getRDMExpectation(Matrix<complexT, Dynamic, Dynamic>& rdm,
                           int orbn, int orb1, int orb2, int orb3, int orb4,
                           Matrix<complexT, 4, 4>& rdmval4,
                           Matrix<complexT, 3, 3>& rdmval3,
                           Matrix<complexT, 2, 2>& rdmval2,
                           vector<int>& rows, vector<int>& cols) {
  //MatrixXcd& rdm, int orbn, int orb1, int orb2, int orb3, int orb4,
  //Matrix4cd& rdmval4, Matrix3cd& rdmval3, Matrix2cd& rdmval2,
  //vector<int>& rows, vector<int>& cols) {


  complexT contribution(0.);
  bool calc3rdm = true;

  
  if (orbn == orb2) {
    rows[0] = orb3; rows[1] = orb4;
    cols[0] = orb1; cols[1] = orbn;
    ResidualGradient::Slice(rdm, rows, cols, rdmval2);
    contribution -= rdmval2.determinant();
    calc3rdm = false;
  }
  if (orbn == orb1) {
    rows[0] = orb3; rows[1] = orb4;
    cols[0] = orb2; cols[1] = orbn;
    ResidualGradient::Slice(rdm, rows, cols, rdmval2);
    contribution += rdmval2.determinant();
    calc3rdm = false;
  }

  if (calc3rdm) {
    rows[0] = orbn; rows[1] = orb3; rows[2] = orb4;
    cols[0] = orbn; cols[1] = orb2; cols[2] = orb1;
    
    ResidualGradient::Slice(rdm, rows, cols, rdmval3);
    contribution =  rdmval3.determinant();
  }
  
  return contribution;
};

//N_orbn N_orbm orb1^dag orb2
template<typename complexT>
complexT getRDMExpectation(Matrix<complexT, Dynamic, Dynamic>& rdm,
                           int orbn, int orbm, int orb1, int orb2,
                           Matrix<complexT, 4, 4>& rdmval4,
                           Matrix<complexT, 3, 3>& rdmval3,
                           Matrix<complexT, 2, 2>& rdmval2,
                           vector<int>& rows, vector<int>& cols) {
  
  rows[0] = orbn; rows[1] = orbm; rows[2] = orb2;
  cols[0] = orbn; cols[1] = orbm; cols[2] = orb1;

  complexT contribution(0.);
  ResidualGradient::Slice(rdm, rows, cols, rdmval3);
  contribution =  rdmval3.determinant();
  
  if (orbn == orb1) {
    rows[0] = orbm; rows[1] = orb2;
    cols[0] = orbm; cols[1] = orbn;
    ResidualGradient::Slice(rdm, rows, cols, rdmval2);
    contribution += rdmval2.determinant();
  }
  if (orbm == orb1) {
    rows[0] = orbn; rows[1] = orb2;
    cols[0] = orbm; cols[1] = orbn;
    ResidualGradient::Slice(rdm, rows, cols, rdmval2);
    contribution -= rdmval2.determinant();
  }
  return contribution;
};



//N_orbn orb1^dag orb2
template<typename complexT>
complexT getRDMExpectation(Matrix<complexT, Dynamic, Dynamic>& rdm,
                           int orbn, int orb1, int orb2,
                           Matrix<complexT, 4, 4>& rdmval4,
                           Matrix<complexT, 3, 3>& rdmval3,
                           Matrix<complexT, 2, 2>& rdmval2,
                           vector<int>& rows, vector<int>& cols) {

  rows[0] = orbn; rows[1] = orb2;
  cols[0] = orbn; cols[1] = orb1;

  complexT contribution(0.);
  ResidualGradient::Slice(rdm, rows, cols, rdmval2);
  contribution =  rdmval2.determinant();
  
  if (orbn == orb1) {
    contribution += rdm(orb2, orbn);
  }

  return contribution;
};


struct Residuals {
  int norbs, nalpha, nbeta;
  MatrixXcd& bra;
  vector<MatrixXcd > ket;
  vector<complex<double> > coeffs;
  double E0;
  int Sz, ngrid;
  
  Residuals(int _norbs, int _nalpha, int _nbeta,
            MatrixXcd& _bra,
            int _ngrid) :  norbs(_norbs),
                           nalpha(_nalpha),
                           nbeta(_nbeta),
                           bra(_bra),
                           ngrid(_ngrid)
  {
    Sz = nalpha-nbeta;
    applyProjector(bra, ket, coeffs, Sz, ngrid);        
  };


  template<typename T>
  T operator()(
      const Matrix<T, Dynamic, 1>& JA) const {

    Matrix<T, Dynamic,1> residue = JA;
    int norbs = Determinant::norbs;
    
    //calculate <bra|P|ket> and
    double detovlp = 0.0;
    MatrixXd mfRDM(2*norbs, 2*norbs); mfRDM.setZero();
    
    for (int g = 0; g<ket.size(); g++) {
      MatrixXcd S = bra.adjoint()*ket[g];
      complex<double> Sdet = S.determinant();
      detovlp += (S.determinant() * coeffs[g]).real();
      mfRDM += ((ket[g] * S.inverse())*bra.adjoint() * coeffs[g] * Sdet).real(); 
    }
    mfRDM = mfRDM/detovlp;
    
    T Energy = 0.;
    Matrix<T, Dynamic, 1> intermediateResidue(2*norbs*(2*norbs+1)/2); intermediateResidue.setZero();

    Matrix<Complex<T>, Dynamic, Dynamic>
        braT(bra.rows(), bra.cols()),
        ketT(bra.rows(), bra.cols());

    for (int i=0; i<bra.rows(); i++)
      for (int j=0; j<bra.cols(); j++)
        braT(i,j) = Complex<T>(bra(i,j).real(), bra(i,j).imag());

    
    for (int g = 0; g<ket.size(); g++)  {

      for (int i=0; i<ket[g].rows(); i++)
        for (int j=0; j<ket[g].cols(); j++)
          ketT(i,j) = Complex<T>(ket[g](i,j).real(), ket[g](i,j).imag());

      Complex<T> coeff = Complex<T>(coeffs[g].real(), coeffs[g].imag());
      Energy += (getResidueSingleKet(JA,
                                     intermediateResidue,
                                     detovlp,
                                     coeff,
                                    braT,
                                     ketT) * coeff).real();
    }    
    
    for (int i=0; i<2*norbs; i++) {
      for (int j=0; j<=i; j++) {
        int index = i*(i+1)/2+j;
        if (i == j) 
          residue(i * (i+1)/2 + j) = intermediateResidue(i*(i+1)/2+j) - (Energy)*mfRDM(i,i);
        else
          residue(i * (i+1)/2 + j) = intermediateResidue(i*(i+1)/2+j) - (Energy)*(mfRDM(j,j)*mfRDM(i,i) - mfRDM(j,i)*mfRDM(i,j));
      }
    }
    
    return residue.squaredNorm();
  }

  template<typename T, typename complexT>
  complexT getResidueSingleKet (
      const Matrix<T, Dynamic, 1>& JA,
      Matrix<T, Dynamic, 1>& residue,
      double detovlp,
      complexT coeff,
      Matrix<complexT, Dynamic, Dynamic>& bra,
      const Matrix<complexT, Dynamic, Dynamic>& ket) const {

    using MatrixXcT = Matrix<Complex<T>, Dynamic, Dynamic>;
    
    int nelec = nalpha+nbeta;
    Matrix<complexT,4,4> rdmResidue4;
    Matrix<complexT,3,3> rdmResidue3;
    Matrix<complexT,2,2> rdmResidue2;
    vector<int> Rows(4), Cols(4);
    
    MatrixXcT LambdaD = bra, LambdaC = ket; //just initializing  
    MatrixXcT S(bra.cols(), ket.cols());

    DiagonalMatrix<T, Dynamic>
        diagcre(2*norbs),
        diagdes(2*norbs);

    complexT Energy(0.);
    //one electron terms
    for (int orb1 = 0; orb1 < 2*norbs; orb1++)
      for (int orb2 = 0; orb2 < 2*norbs; orb2++) {
        
        double integral = I1( (orb1%norbs) * 2 + orb1/norbs, (orb2%norbs) * 2 + orb2/norbs);
        
        if (abs(integral) < schd.epsilon) continue;
        
        complexT factor = getCreDesDiagMatrix(diagcre, diagdes, orb1, orb2, norbs, JA);
        LambdaD = diagdes*ket;
        LambdaC = diagcre*bra;
        S = LambdaC.adjoint()*LambdaD;
        
        complexT Sdet = S.determinant();
        MatrixXcT rdm = (LambdaD * S.inverse())*LambdaC.adjoint();
      
        factor *= Sdet/detovlp;
        Energy += rdm(orb2, orb1) * integral * factor;

        complexT res;
        for (int orbn = 0; orbn < 2*norbs; orbn++) {
          for (int orbm = 0; orbm < orbn; orbm++) {
            res = ResidualGradient::getRDMExpectation(rdm, orbn, orbm, orb1, orb2, rdmResidue4, rdmResidue3, rdmResidue2, Rows, Cols);
            residue(orbn*(orbn+1)/2 + orbm) += (res * factor * integral * coeff).real();
          }
          res = ResidualGradient::getRDMExpectation(rdm, orbn, orb1, orb2, rdmResidue4, rdmResidue3, rdmResidue2, Rows, Cols);
          residue(orbn*(orbn+1)/2 + orbn) += (res * factor * integral * coeff).real();
        }
        
      }
    
    
  //one electron terms
  for (int orb1 = 0; orb1 < 2*norbs; orb1++)
    for (int orb2 = 0; orb2 < orb1; orb2++) {
      for (int orb3 = 0; orb3 < 2*norbs; orb3++)
        for (int orb4 = 0; orb4 < orb3; orb4++) {
            
          int Orb1 = (orb1%norbs)* 2 + orb1/norbs;
          int Orb2 = (orb2%norbs)* 2 + orb2/norbs;
          int Orb3 = (orb3%norbs)* 2 + orb3/norbs;
          int Orb4 = (orb4%norbs)* 2 + orb4/norbs;
            
          double integral = (I2(Orb1, Orb4, Orb2, Orb3) - I2(Orb2, Orb4, Orb1, Orb3));
            
          if (abs(integral) < schd.epsilon) continue;
            
          complexT factor = getCreDesDiagMatrix(diagcre, diagdes, orb1, orb2, orb3, orb4, norbs, JA);
          LambdaD = diagdes*ket;
          LambdaC = diagcre*bra;
          S = LambdaC.adjoint()*LambdaD;

          /*
          Eigen::FullPivLU<MatrixXcd> lu(S);
          MatrixXcd rdm = LambdaD * lu.solve(LambdaC.adjoint());
          complex<double> Sdet = 1.0;
          for (int i=0; i<nelec; i++)
            Sdet *= lu.matrixLU()(i,i);
          */
          complexT Sdet = S.determinant();
          MatrixXcT rdm = (LambdaD * S.inverse())*LambdaC.adjoint();
          
          complexT rdmval = rdm(orb4, orb1) * rdm(orb3, orb2) - rdm(orb3, orb1) * rdm(orb4, orb2);
            
          factor *= Sdet/detovlp;

          
          Energy += rdmval * integral * factor;

          complexT res;
          for (int orbn = 0; orbn < 2*norbs; orbn++) {
            for (int orbm = 0; orbm < orbn; orbm++) {
              res = ResidualGradient::getRDMExpectation(rdm, orbn, orbm, orb1, orb2, orb3, orb4, rdmResidue4, rdmResidue3,
                                      rdmResidue2, Rows, Cols);
              residue(orbn*(orbn+1)/2 + orbm) += (res*factor*integral * coeff).real();
            }
            res = ResidualGradient::getRDMExpectation(rdm, orbn, orb1, orb2, orb3, orb4, rdmResidue4, rdmResidue3,
                                    rdmResidue2, Rows, Cols);
            residue(orbn*(orbn+1)/2 + orbn) += (res*factor*integral * coeff).real();
          }
        }
    }

    return Energy;
  }


};

};


