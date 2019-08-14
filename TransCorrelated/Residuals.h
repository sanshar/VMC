#ifndef EvalRESIDUALS_HEADER_H
#define EvalRESIDUALS_HEADER_H
#include <Eigen/Dense>
#include <vector>
#include "Complex.h"
using namespace std;

using namespace Eigen;
using DiagonalXd = Eigen::DiagonalMatrix<double, Eigen::Dynamic>;

int index(int I, int J) ;

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


struct Residuals {
  int norbs, nalpha, nbeta;
  MatrixXcd& bra;
  vector<MatrixXcd > ket;
  vector<complex<double> > coeffs;
  double E0;
  VectorXd& Jastrow;
  int Sz, ngrid;
  
  Residuals(int _norbs, int _nalpha, int _nbeta,
            VectorXd& _Jastrow,
            MatrixXcd& _bra,
            int _ngrid) :  norbs(_norbs),
                           nalpha(_nalpha),
                           nbeta(_nbeta),
                           Jastrow(_Jastrow),
                           bra(_bra),
                           ngrid(_ngrid)
  {
    Sz = nalpha-nbeta;
    applyProjector(bra, ket, coeffs, Sz, ngrid);        
  };


  void updateOrbitals(VectorXd& braReal);
  int getOrbitalResidue(
      const VectorXd& braGiven,
      VectorXd& residue);
  
  void getOrbResidueSingleKet(
      double detovlp,
      MatrixXcd& bra,
      MatrixXcd& ket,
      complex<double> coeff,
      MatrixXcd& residue) ;

  int getJastrowResidue(
      const VectorXd& JA,
      VectorXd& residue);
  
  complex<double> getResidueSingleKet (
      const VectorXd& JA,
      VectorXd& residue, double detovlp,
      complex<double> coeff,
      MatrixXcd& bra,
      const MatrixXcd& ket) const;

  //obviously the energy should be real
  double Energy() const;
  complex<double> energyContribution(
      double &detovlp,
      const MatrixXcd& bra,
      const MatrixXcd& ket) const;

  template<typename T>
  T getOvlp(const Matrix<T, Dynamic, 1>& braReal) const
  {
    Matrix<Complex<T>, Dynamic, Dynamic> braVar(bra.rows(), bra.cols());
    for (int i=0; i<2*norbs; i++) 
      for (int j=0; j<bra.cols(); j++)  {
        braVar(i,j) = Complex<T>(braReal(2*(i*bra.cols()+j)), braReal(2*(i*bra.cols()+j)+1));
      }
    
    Matrix<Complex<T>, Dynamic, Dynamic> S, ketT(bra.rows(), bra.cols());
    Complex<T> detovlp;
    for (int g = 0; g<ket.size(); g++) {

      for (int i=0; i<ket[g].rows(); i++)
        for (int j=0; j<ket[g].cols(); j++)
          ketT(i,j) = Complex<T>(ket[g](i,j).real(), ket[g](i,j).imag());
      
      S = braVar.adjoint()*ketT;
      Complex<T> Sdet = S.determinant();
      Complex<T> coeffg = Complex<T>(coeffs[g].real(), coeffs[g].imag());
      Complex<T> detcoeff = Sdet*coeffg;
      detovlp += detcoeff;
    }

    return detovlp.real();
  }

  template<typename T>
  T operator()(Matrix<T, Dynamic, 1>& braReal) const
  {
    //calculate <bra|P|ket> and
    
    MatrixXcd S;
    double detovlp;
    for (int g = 0; g<ket.size(); g++) {
      S = bra.adjoint()*ket[g];
      detovlp += (S.determinant() * coeffs[g]).real();
    }

    Matrix<Complex<T>, Dynamic, Dynamic> braVar(bra.rows(), bra.cols());
    for (int i=0; i<2*norbs; i++) 
      for (int j=0; j<bra.cols(); j++) 
        braVar(i,j) = Complex<T>(braReal(2*(i*bra.cols()+j)), braReal(2*(i*bra.cols()+j)+1));

    
    T Energy = 0.0;
    for (int g = 0; g<ket.size(); g++) {
      //auto energy = gradEnergyContribution(detovlp, braVar, ket[g]);
      //Energy += (energy * coeffs[g]).real();
      Complex<T> coeff(coeffs[g].real(), coeffs[g].imag());
      Energy += (gradEnergyContribution(detovlp, braVar, ket[g]) * coeff).real();
    }
    
    const_cast<double&>(this->E0) = Energy.val();
    return Energy;
  }

  // to obtain stan derivative
  template<typename T>
  Complex<T> gradEnergyContribution(
      double &detovlp,
      const Matrix<Complex<T>, Dynamic, Dynamic>& braVar,
      const MatrixXcd& ket) const
  {
    Matrix<Complex<T>, Dynamic, Dynamic> LambdaD, ketT(ket.rows(), ket.cols());
    for (int i=0; i<ket.rows(); i++)
      for (int j=0; j<ket.cols(); j++)
        ketT(i,j) = Complex<T>(ket(i,j).real(), ket(i,j).imag());

    Matrix<Complex<T>, Dynamic, Dynamic> LambdaC, S;
    DiagonalMatrix<Complex<T>, Dynamic> diagcreVar(2*norbs), diagdesVar(2*norbs);
    
    DiagonalXd diagcre(2*norbs),
        diagdes(2*norbs);
    
    Complex<T> Energy ;
    //one electron terms
    for (int orb1 = 0; orb1 < 2*norbs; orb1++)
      for (int orb2 = 0; orb2 < 2*norbs; orb2++) {
        
        double integral = I1( (orb1%norbs) * 2 + orb1/norbs, (orb2%norbs) * 2 + orb2/norbs);
        
        if (abs(integral) < schd.epsilon) continue;
        
        double factor = getCreDesDiagMatrix(diagcre, diagdes, orb1, orb2, norbs, Jastrow);
        for (int i=0; i<2*norbs; i++) {
          diagcreVar.diagonal()[i] = Complex<T>(diagcre.diagonal()[i]);
          diagdesVar.diagonal()[i] = Complex<T>(diagdes.diagonal()[i]);
        }
        
        LambdaD = diagdesVar*ketT;
        LambdaC = diagcreVar*braVar;
        
        S = LambdaC.adjoint()*LambdaD;
        auto Sinv = S.inverse();

        //factor *= S.determinant()/detovlp;
        //**don't need to calculate the entire RDM, should make it more efficient
        Complex<T> rdm = ((LambdaD.row(orb2) * Sinv)
                          *LambdaC.adjoint().col(orb1))(0,0);
        Energy += rdm * integral * factor * S.determinant() / detovlp;
        //Energy += rdm * integral * S.determinant() / detovlp;
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
            
            double factor = getCreDesDiagMatrix(diagcre, diagdes, orb1, orb2, orb3, orb4, norbs, Jastrow);
            for (int i=0; i<2*norbs; i++) {
              diagcreVar.diagonal()[i] = Complex<T>(diagcre.diagonal()[i]);
              diagdesVar.diagonal()[i] = Complex<T>(diagdes.diagonal()[i]);
            }
            
            LambdaD = diagdesVar*ketT;
            LambdaC = diagcreVar*braVar;
            //LambdaC = diagcre*braVar;
            
            S = LambdaC.adjoint()*LambdaD;
            Matrix<Complex<T>, Dynamic, Dynamic> Sinv = S.inverse();
            
            Complex<T> rdm1 = ((LambdaD.row(orb4) * Sinv) * LambdaC.adjoint().col(orb1));
            Complex<T> rdm2 = ((LambdaD.row(orb3) * Sinv) * LambdaC.adjoint().col(orb2));
            Complex<T> rdm3 = ((LambdaD.row(orb3) * Sinv) * LambdaC.adjoint().col(orb1));
            Complex<T> rdm4 = ((LambdaD.row(orb4) * Sinv) * LambdaC.adjoint().col(orb2));
            
            Energy += (rdm1*rdm2 - rdm3*rdm4) * integral * factor * S.determinant() / detovlp;
          }
      }
    
    return Energy;
  }

  

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


#endif
