#include "Residuals.h"
#include "Determinants.h"
#include "global.h"
#include "integral.h"
#include "input.h"

void applyProjector(
    MatrixXcd& bra,
    vector<MatrixXcd>& ketvec,
    vector<complex<double> >& coeffs,
    int Sz,
    int ngrid) {

  int norbs = Determinant::norbs;
  double m = 1.*Sz;

  complex<double> iImag(0, 1.0);
  coeffs.resize(ngrid, 0.0);
  ketvec.resize(ngrid, bra);

  for (int g = 0; g<ngrid; g++) {
    DiagonalMatrix<complex<double>, Dynamic> phi(2*norbs);
    double angle = g*2*M_PI/ngrid;
    for (int i=0; i<norbs; i++) {
      phi.diagonal()[i]       = exp(iImag*angle/2.);
      phi.diagonal()[i+norbs] = exp(-iImag*angle/2.);
    }
    ketvec[g] = phi * bra;
    coeffs[g] = exp(-iImag*angle*m)/ngrid;
  }
}


void fillJastrowfromWfn(MatrixXd& Jtmp, VectorXd& JA) {
  int norbs = Determinant::norbs;
  
  for (int i=0; i<2*norbs; i++) {
    int I = (i/2) + (i%2)*norbs;
    for (int j=0; j<i; j++) {
      int J = (j/2) + (j%2)*norbs;
      
      JA(index(J, I)) = log(Jtmp(i, j))/2.0;
    }
    JA(index(I, I)) = log(Jtmp(i,i));///2;
  }
  
}

void fillWfnfromJastrow(VectorXd& JA, MatrixXd& Jtmp) {
  int norbs = Determinant::norbs;
  
  for (int i=0; i<2*norbs; i++) {
    int I = (i/2) + (i%2)*norbs;
    for (int j=0; j<i; j++) {
      int J = (j/2) + (j%2)*norbs;
      
      Jtmp(i, j) = exp(2.0 * JA(index(J, I)));
    }
    Jtmp(i, i) = exp(JA(index(I, I)));
  }
  
}

void fillWfnOrbs(MatrixXcd& orbitals, VectorXd& variables) {
  int norbs  = Determinant::norbs;
  int nalpha = Determinant::nalpha;
  int nbeta  = Determinant::nbeta;
  int nelec =  nalpha + nbeta;

  //Use the rotation matrix variables and update the bra
  MatrixXcd U(2*norbs-nelec, nelec);
  for (int a=0; a<2*norbs-nelec; a++) 
    for (int i=0; i<nelec; i++) 
      U(a, i) = complex<double>( variables( 2* (a*nelec+i)  ),
                                 variables( 2* (a*nelec+i)+1)
                                 );

  orbitals.block(0, 0, 2*norbs, nelec) += orbitals.block(0, nelec, 2*norbs, 2*norbs-nelec) * U;
}

double GetResidual::getResidue(const VectorXd& variables,
                               VectorXd& residual,
                               bool getJastrowResidue,
                               bool getOrbitalResidue) {

  int norbs  = Determinant::norbs;
  int nalpha = Determinant::nalpha;
  int nbeta  = Determinant::nbeta;
  int nelec =  nalpha + nbeta;
  
  int nJastrowVars = 2*norbs*(2*norbs+1)/2;
  int nOrbitalVars = 2*norbs*(nalpha+nbeta);
  VectorXd Jastrow = variables.block(0,0,nJastrowVars, 1);


  //Use the rotation matrix variables and update the bra
  MatrixXcd U(2*norbs-nelec, nelec);
  for (int a=0; a<2*norbs-nelec; a++) 
    for (int i=0; i<nelec; i++) 
      U(a, i) = complex<double>( variables( nJastrowVars + 2* (a*nelec+i)  ),
                                 variables( nJastrowVars + 2* (a*nelec+i)+1)
                                 );

  MatrixXcd phiupdate = orbitals.block(0, nelec, 2*norbs, 2*norbs-nelec) * U;
  MatrixXcd bra = phiupdate + orbitals.block(0,0,2*norbs, nelec);


  //store the jastrow and orbital residue
  VectorXd JastrowResidue = 0 * Jastrow;
  VectorXd braVarsResidue( 2*(2*norbs - nelec)* nelec); braVarsResidue.setZero();

  //apply the Sz projector and generate a linear combination of kets
  vector<MatrixXcd> ket; vector<complex<double>> coeffs;
  applyProjector(bra, ket, coeffs, nalpha-nbeta, ngrid); 


  MatrixXcd S;
  double detovlp = 0.;

  //these terms are needed for calculate the orbital and jastrow gradient respectively
  MatrixXcd orbitalOverlapGrad = bra; orbitalOverlapGrad.setZero();
  VectorXd NiNjRDM(2*norbs*(2*norbs+1)/2); NiNjRDM.setZero();
  MatrixXcd mfRDM(2*norbs, 2*norbs);

  
  for (int g = 0; g<ket.size(); g++) {
    S = bra.adjoint()*ket[g];
    complex<double> Sdet = S.determinant();
    detovlp += (Sdet * coeffs[g]).real();

    if (getOrbitalResidue)
      orbitalOverlapGrad += Sdet * coeffs[g] * ket[g] * S.inverse();

    if (getJastrowResidue) {
      mfRDM = ((ket[g] * S.inverse())*bra.adjoint());     
      for (int i=0; i<2*norbs; i++) 
        for (int j=0; j<=i; j++) {
          int index = i*(i+1)/2+j;
          NiNjRDM(index) +=
              i == j ?
              (mfRDM(j, i)* conj(coeffs[g]) * Sdet).real() :
              ((mfRDM(j,j)*mfRDM(i,i) - mfRDM(i,j) * mfRDM(j,i))* conj(coeffs[g]) * Sdet).real();
        }
    }

  }
  orbitalOverlapGrad /= detovlp;
  NiNjRDM /= detovlp;


  
  double Energy = 0.0;
  JastrowResidue.setZero(); braVarsResidue.setZero();

  MatrixXcd braResidueMat = 0.*bra; braResidueMat.setZero();
  //for (int g = 0; g<ket.size(); g++)  {
    Energy += getResidueSingleKet(detovlp,
                                  bra, ket, coeffs, braResidueMat,
                                  Jastrow, JastrowResidue,
                                  getJastrowResidue, getOrbitalResidue) ;
    //}  

  if (getJastrowResidue)
    JastrowResidue -= Energy *NiNjRDM;
  
  if (getOrbitalResidue) {
    braResidueMat -= Energy * orbitalOverlapGrad;
    
    
    MatrixXcd nonRedundantOrbResidue =
        orbitals.block(0,nelec, 2*norbs, 2*norbs - nelec).adjoint()*braResidueMat;
    
    int nelec = bra.cols();
    for (int i=0; i<2*norbs - nelec; i++)
      for (int j=0; j<nelec; j++) {
        braVarsResidue(2 * (i * nelec + j)    ) = nonRedundantOrbResidue(i,j).real();
        braVarsResidue(2 * (i * nelec + j) + 1) = nonRedundantOrbResidue(i,j).imag();
      }
    
    
  }
  
  residual.block(0, 0, nJastrowVars, 1) = JastrowResidue;
  residual.block(nJastrowVars, 0, braVarsResidue.size(), 1) = braVarsResidue;
  
  return Energy+coreE;
}

double GetResidual::getJastrowResidue(const VectorXd& jastrowvars,
                                     const VectorXd& braVars,
                                     VectorXd& jastrowResidue) {
  int nJastrowVars = jastrowvars.size(),
      nBraVars = braVars.size();
  VectorXd variables(nJastrowVars+nBraVars);
  
  variables.block(0, 0, nJastrowVars, 1) = jastrowvars;
  variables.block(nJastrowVars, 0, nBraVars, 1) = braVars;
  
  VectorXd residue = variables;
  double Energy = getResidue(variables, residue, true, false);

  jastrowResidue = residue.block(0, 0, nJastrowVars, 1);
  return Energy;
}

double GetResidual::getOrbitalResidue(const VectorXd& jastrowvars,
                                      const VectorXd& braVars,
                                      VectorXd& orbitalResidue) {

  int nJastrowVars = jastrowvars.size(),
      nBraVars = braVars.size();
  VectorXd variables(nJastrowVars+nBraVars);
  
  variables.block(0, 0, nJastrowVars, 1) = jastrowvars;
  variables.block(nJastrowVars, 0, nBraVars, 1) = braVars;

  VectorXd residue = variables;
  double Energy = getResidue(variables, residue, false, true);

  orbitalResidue = residue.block(nJastrowVars, 0, nBraVars, 1);
  return Energy;
}

double GetResidual::getResidueSingleKet(
    double detovlp,
    MatrixXcd& bra,
    vector<MatrixXcd>& ketvec,
    vector<complex<double>>& coeffvec,
    MatrixXcd& orbitalResidueMat,
    VectorXd& Jastrow,
    VectorXd& JastrowResidue,
    bool getJastrowResidue,
    bool getOrbitalResidue) {

  int norbs  = Determinant::norbs;
  int nalpha = Determinant::nalpha;
  int nbeta  = Determinant::nbeta;

  int nelec = nalpha+nbeta;
  Matrix4cd rdmResidue4;
  Matrix3cd rdmResidue3;
  Matrix2cd rdmResidue2;
  vector<int> Rows(4), Cols(4);

  MatrixXcd LambdaD, LambdaC, S;

  DiagonalXd diagcre(2*norbs),
      diagdes(2*norbs);

  //MatrixXcd orbitalResidueMat = bra; orbitalResidueMat.setZero();
  
  MatrixXcd rdm;
  complex<double> Energy;

  size_t ketSizeOrb = ketvec.size() * 2 * norbs;
  //one electron terms
  //for (int g=0; g<ketvec.size(); g++) {
  //MatrixXcd& ket = ketvec[g];
  //complex<double> coeff = coeffvec[g];
  //for (int orb1 = 0; orb1 < 2*norbs; orb1++)

  for (int gorb1 = commrank; gorb1 < ketSizeOrb; gorb1+=commsize) {
    int g = gorb1 / (2 * norbs), orb1 = gorb1 % (2 * norbs);
    MatrixXcd& ket = ketvec[g];
    complex<double> coeff = coeffvec[g];

    for (int orb2 = 0; orb2 < 2*norbs; orb2++) {
        
      double integral = I1( (orb1%norbs) * 2 + orb1/norbs, (orb2%norbs) * 2 + orb2/norbs);
        
      if (abs(integral) < schd.epsilon) continue;
        
      complex<double> factor = getCreDesDiagMatrix(diagcre, diagdes, orb1, orb2, norbs, Jastrow);
      LambdaD = diagdes*ket;
      LambdaC = diagcre*bra;
        
      S = LambdaC.adjoint()*LambdaD;


      std::complex<double> Sdet = S.determinant();
      MatrixXcd Sinv = S.inverse();

      if (getJastrowResidue) rdm = (LambdaD * Sinv) * LambdaC.adjoint();

      complex<double> rdmval = getJastrowResidue ?
          rdm(orb2, orb1) : ((LambdaD.row(orb2) * Sinv)*LambdaC.adjoint().col(orb1))(0,0);

      factor *= integral * Sdet * coeff / detovlp;

      if (getOrbitalResidue) {
        MatrixXcd JJphi = diagcre*LambdaD;
        MatrixXcd JJphiSinv = JJphi*Sinv;
        
        orbitalResidueMat.row(orb1) += factor * ( diagcre.diagonal()[orb1] * (LambdaD.row(orb2) * Sinv) );
        orbitalResidueMat += factor *
            (JJphiSinv * rdmval
             -((JJphiSinv*LambdaC.adjoint().col(orb1)) * (LambdaD.row(orb2) * Sinv)));
      }
      
      Energy += rdmval * factor ;
      //Energy += rdm * integral * S.determinant() / detovlp;

      if (getJastrowResidue) {
        complex<double> res;
        for (int orbn = 0; orbn < 2*norbs; orbn++) {
          for (int orbm = 0; orbm < orbn; orbm++) {
            res = getRDMExpectation(rdm, orbn, orbm, orb1, orb2,
                                    rdmResidue4, rdmResidue3, rdmResidue2, Rows, Cols);
            JastrowResidue(orbn*(orbn+1)/2 + orbm) += (res * factor).real();
          }
          res = getRDMExpectation(rdm, orbn, orb1, orb2,
                                  rdmResidue4, rdmResidue3, rdmResidue2, Rows, Cols);
          JastrowResidue(orbn*(orbn+1)/2 + orbn) += (res * factor).real();
        }
      }
      
    }
    

  //one electron terms
    //for (int orb1 = 0; orb1 < 2*norbs; orb1++)
    for (int orb2 = 0; orb2 < orb1; orb2++) {
      for (int orb3 = 0; orb3 < 2*norbs; orb3++)
        for (int orb4 = 0; orb4 < orb3; orb4++) {
            
          int Orb1 = (orb1%norbs)* 2 + orb1/norbs;
          int Orb2 = (orb2%norbs)* 2 + orb2/norbs;
          int Orb3 = (orb3%norbs)* 2 + orb3/norbs;
          int Orb4 = (orb4%norbs)* 2 + orb4/norbs;
            
          double integral = (I2(Orb1, Orb4, Orb2, Orb3) - I2(Orb2, Orb4, Orb1, Orb3));
            
          if (abs(integral) < schd.epsilon) continue;
            
          complex<double> factor = getCreDesDiagMatrix(diagcre, diagdes, orb1, orb2, orb3, orb4, norbs, Jastrow);
          LambdaD = diagdes*ket;
          LambdaC = diagcre*bra;
          S = LambdaC.adjoint()*LambdaD;
          complex<double> Sdet = S.determinant();
          MatrixXcd Sinv = S.inverse();

          if (getJastrowResidue) 
            rdm = (LambdaD * S.inverse())*LambdaC.adjoint();

          //**don't need to calculate the entire RDM, should make it more efficient
          complex<double> rdm1 = getJastrowResidue ?
              rdm(orb4, orb1) : ((LambdaD.row(orb4) * Sinv) * LambdaC.adjoint().col(orb1));
          complex<double> rdm2 = getJastrowResidue ?
              rdm(orb3, orb2) : ((LambdaD.row(orb3) * Sinv) * LambdaC.adjoint().col(orb2));
          complex<double> rdm3 = getJastrowResidue ?
              rdm(orb3, orb1) : ((LambdaD.row(orb3) * Sinv) * LambdaC.adjoint().col(orb1));
          complex<double> rdm4 = getJastrowResidue ?
              rdm(orb4, orb2) : ((LambdaD.row(orb4) * Sinv) * LambdaC.adjoint().col(orb2));

          factor *= integral * Sdet * coeff/ detovlp;

          if (getOrbitalResidue) {
            MatrixXcd JJphi = diagcre*LambdaD;
            MatrixXcd JJphiSinv = JJphi*Sinv;
            
            orbitalResidueMat.row(orb2) += factor * rdm1
                * ( (diagcre.diagonal()[orb2]) * (LambdaD.row(orb3) * Sinv) );
            
            orbitalResidueMat.row(orb1) += factor * rdm2
                * ( (diagcre.diagonal()[orb1]) * (LambdaD.row(orb4) * Sinv) );
            
            orbitalResidueMat.row(orb2) -= factor * rdm3
                * ( (diagcre.diagonal()[orb2]) * (LambdaD.row(orb4) * Sinv) );
            
            orbitalResidueMat.row(orb1) -= factor * rdm4
                * ( (diagcre.diagonal()[orb1]) * (LambdaD.row(orb3) * Sinv) );
            
            orbitalResidueMat += factor * rdm1 *
                (- ( (JJphiSinv*LambdaC.adjoint().col(orb2)) * (LambdaD.row(orb3) * Sinv) ));
            
            orbitalResidueMat += factor * rdm2 *
                (- ( (JJphiSinv*LambdaC.adjoint().col(orb1)) * (LambdaD.row(orb4) * Sinv) ));
            
            orbitalResidueMat -= factor * rdm3 *
                (- ( (JJphiSinv*LambdaC.adjoint().col(orb2)) * (LambdaD.row(orb4) * Sinv) ));
            
            orbitalResidueMat -= factor * rdm4 *
                (- ( (JJphiSinv*LambdaC.adjoint().col(orb1)) * (LambdaD.row(orb3) * Sinv) ));
            
            orbitalResidueMat += factor*(rdm1*rdm2 - rdm3*rdm4) * JJphiSinv;
            
          }
          Energy += (rdm1*rdm2 - rdm3*rdm4) * factor ;

          if (getJastrowResidue) {
            complex<double> res;
            for (int orbn = 0; orbn < 2*norbs; orbn++) {
              for (int orbm = 0; orbm < orbn; orbm++) {
                res = getRDMExpectation(rdm, orbn, orbm, orb1, orb2, orb3, orb4,
                                        rdmResidue4, rdmResidue3, rdmResidue2, Rows, Cols);
                JastrowResidue(orbn*(orbn+1)/2 + orbm) += (res*factor).real();
              }
              res = getRDMExpectation(rdm, orbn, orb1, orb2, orb3, orb4, rdmResidue4, rdmResidue3,
                                      rdmResidue2, Rows, Cols);
              JastrowResidue(orbn*(orbn+1)/2 + orbn) += (res*factor).real();
            }
          }
        }
    }
  }
#ifndef SERIAL
  size_t jsize = JastrowResidue.size();
  MPI_Allreduce(MPI_IN_PLACE, &JastrowResidue(0), jsize, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  size_t osize = 2*orbitalResidueMat.rows() * orbitalResidueMat.cols();
  MPI_Allreduce(MPI_IN_PLACE, &orbitalResidueMat(0,0), osize, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  size_t two = 2;
  MPI_Allreduce(MPI_IN_PLACE, &Energy, two, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  return Energy.real();
}

      


int index(int I, int J) {
  return max(I,J)*(max(I,J)+1)/2 + min(I,J);
}


//term is orb1^dag orb2
double getCreDesDiagMatrix(DiagonalXd& diagcre,
                           DiagonalXd& diagdes,
                           int orb1,
                           int orb2,
                           int norbs,
                           const VectorXd&JA) {
  
  for (int j=0; j<2*norbs; j++) {
    diagcre.diagonal()[j] = exp(-2.*JA(index(orb1, j)));
    diagdes.diagonal()[j] = exp( 2.*JA(index(orb2, j)));
  }

  double factor = exp( JA(index(orb1, orb1)) - JA(index(orb2, orb2)));
  return factor;
}

//term is orb1^dag orb2^dag orb3 orb4
double getCreDesDiagMatrix(DiagonalXd& diagcre,
                           DiagonalXd& diagdes,
                           int orb1,
                           int orb2,
                           int orb3,
                           int orb4,
                           int norbs,
                           const VectorXd&JA) {

  for (int j=0; j<2*norbs; j++) {
    diagcre.diagonal()[j] = exp(-2.*JA(index(orb1, j)) - 2.*JA(index(orb2, j)));
    diagdes.diagonal()[j] = exp(+2.*JA(index(orb3, j)) + 2.*JA(index(orb4, j)));
  }

  double factor = exp( 2*JA(index(orb1, orb2)) + JA(index(orb1, orb1)) + JA(index(orb2, orb2))
                       -2*JA(index(orb3, orb4)) - JA(index(orb3, orb3)) - JA(index(orb4, orb4))) ;
  return factor;
}



//N_orbn N_orbm orb1^dag orb2^dag orb3 orb4
complex<double> getRDMExpectation(MatrixXcd& rdm, int orbn, int orbm,
                                  int orb1, int orb2, int orb3, int orb4,
                                  Matrix4cd& rdmval4, Matrix3cd& rdmval3, Matrix2cd& rdmval2,
                                  vector<int>& rows, vector<int>& cols) {

  complex<double> contribution = 0;
  bool calc4rdm = true;

  if (orbm == orb2) {
    rows[0] = orbn; rows[1] = orb3; rows[2] = orb4;
    cols[0] = orb1; cols[1] = orbm; cols[2] = orbn;
    Slice(rdm, rows, cols, rdmval3);
    contribution -= rdmval3.determinant();
    calc4rdm = false;
  }
  if (orbm == orb1) {
    rows[0] = orbn; rows[1] = orb3; rows[2] = orb4;
    cols[0] = orb2; cols[1] = orbm; cols[2] = orbn;
    Slice(rdm, rows, cols, rdmval3);
    contribution += rdmval3.determinant();
    calc4rdm = false;
  }
  if (orbn == orb2) {
    rows[0] = orbm; rows[1] = orb3; rows[2] = orb4;
    cols[0] = orb1; cols[1] = orbm; cols[2] = orbn;
    Slice(rdm, rows, cols, rdmval3);
    contribution += rdmval3.determinant();
    calc4rdm = false;
  }
  if (orbn == orb2 && orbm == orb1) {
    rows[0] = orb3; rows[1] = orb4;
    cols[0] = orbm; cols[1] = orbn;
    Slice(rdm, rows, cols, rdmval2);
    contribution -= rdmval2.determinant();
    calc4rdm = false;
  }
  if (orbn == orb1) {
    rows[0] = orbm; rows[1] = orb3; rows[2] = orb4;
    cols[0] = orb2; cols[1] = orbm; cols[2] = orbn;
    Slice(rdm, rows, cols, rdmval3);
    contribution -= rdmval3.determinant();
    calc4rdm = false;
  }
  if (orbn == orb1 && orbm == orb2) {
    rows[0] = orb3; rows[1] = orb4;
    cols[0] = orbm; cols[1] = orbn;
    Slice(rdm, rows, cols, rdmval2);
    contribution += rdmval2.determinant();
    calc4rdm = false;
  }

  if (calc4rdm) {
    rows[0] = orbm; rows[1] = orbn; rows[2] = orb3; rows[3] = orb4;
    cols[0] = orbm; cols[1] = orbn; cols[2] = orb2; cols[3] = orb1;
    
    Slice(rdm, rows, cols, rdmval4);
    contribution =  rdmval4.determinant();
  }
  
  return contribution;
}


//N_orbn orb1^dag orb2^dag orb3 orb4
complex<double> getRDMExpectation(MatrixXcd& rdm, int orbn, int orb1, int orb2, int orb3, int orb4,
                           Matrix4cd& rdmval4, Matrix3cd& rdmval3, Matrix2cd& rdmval2,
                           vector<int>& rows, vector<int>& cols) {


  complex<double> contribution = 0;
  bool calc3rdm = true;

  
  if (orbn == orb2) {
    rows[0] = orb3; rows[1] = orb4;
    cols[0] = orb1; cols[1] = orbn;
    Slice(rdm, rows, cols, rdmval2);
    contribution -= rdmval2.determinant();
    calc3rdm = false;
  }
  if (orbn == orb1) {
    rows[0] = orb3; rows[1] = orb4;
    cols[0] = orb2; cols[1] = orbn;
    Slice(rdm, rows, cols, rdmval2);
    contribution += rdmval2.determinant();
    calc3rdm = false;
  }

  if (calc3rdm) {
    rows[0] = orbn; rows[1] = orb3; rows[2] = orb4;
    cols[0] = orbn; cols[1] = orb2; cols[2] = orb1;
    
    Slice(rdm, rows, cols, rdmval3);
    contribution =  rdmval3.determinant();
  }
  
  return contribution;
}

//N_orbn N_orbm orb1^dag orb2
complex<double> getRDMExpectation(const MatrixXcd& rdm, int orbn, int orbm, int orb1, int orb2,
                           Matrix4cd& rdmval4, Matrix3cd& rdmval3, Matrix2cd& rdmval2,
                           vector<int>& rows, vector<int>& cols) {
  
  rows[0] = orbn; rows[1] = orbm; rows[2] = orb2;
  cols[0] = orbn; cols[1] = orbm; cols[2] = orb1;

  complex<double> contribution;
  Slice(rdm, rows, cols, rdmval3);
  contribution =  rdmval3.determinant();
  
  if (orbn == orb1) {
    rows[0] = orbm; rows[1] = orb2;
    cols[0] = orbm; cols[1] = orbn;
    Slice(rdm, rows, cols, rdmval2);
    contribution += rdmval2.determinant();
  }
  if (orbm == orb1) {
    rows[0] = orbn; rows[1] = orb2;
    cols[0] = orbm; cols[1] = orbn;
    Slice(rdm, rows, cols, rdmval2);
    contribution -= rdmval2.determinant();
  }
  return contribution;
}



//N_orbn orb1^dag orb2
complex<double> getRDMExpectation(MatrixXcd& rdm, int orbn, int orb1, int orb2,
                           Matrix4cd& rdmval4, Matrix3cd& rdmval3, Matrix2cd& rdmval2,
                           vector<int>& rows, vector<int>& cols) {

  rows[0] = orbn; rows[1] = orb2;
  cols[0] = orbn; cols[1] = orb1;

  complex<double> contribution ;
  Slice(rdm, rows, cols, rdmval2);
  contribution =  rdmval2.determinant();
  
  if (orbn == orb1) {
    contribution += rdm(orb2, orbn);
  }

  return contribution;
}






