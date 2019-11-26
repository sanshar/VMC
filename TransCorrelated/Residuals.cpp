#include <algorithm>
#include "Residuals.h"
#include "Determinants.h"
#include "global.h"
#include "integral.h"
#include "input.h"
#include "calcRDM.h"
#include <functional>

using namespace std;
using namespace std::placeholders;

//this is just theSz projector
void applyProjector(
    MatrixXcd& bra,
    vector<MatrixXcd>& ketvec,
    vector<complex<double> >& coeffs,
    int Sz,
    int ngrid) {

  int norbs = Determinant::norbs;
  double m = 1.*Sz/2.;
  int count = 0;
  complex<double> iImag(0, 1.0);

  auto CoeffF =  [&, count](complex<double>&) mutable -> complex<double>
      {return exp(- iImag * ((++count) - 1) * 2 * M_PI * m/ngrid)/ngrid;};

  auto ketF =    [&, count](MatrixXcd&) mutable -> MatrixXcd
      { VectorXcd phi = VectorXcd::Ones(2*norbs);
        phi.segment(0,norbs)     *= exp(iImag * count * 2 * M_PI / ngrid /2.);
        phi.segment(norbs,norbs) *= exp(-iImag * count * 2 * M_PI / ngrid /2.);
        count ++;
        return phi.asDiagonal()*bra;
      };

  coeffs.resize(ngrid);
  std::transform(coeffs.begin(), coeffs.end(), coeffs.begin(),
                 CoeffF);


  ketvec.resize(ngrid);
  std::transform(ketvec.begin(), ketvec.end(), ketvec.begin(),
                 ketF);
  
}



MatrixXcd fillWfnOrbs(MatrixXcd& orbitals, VectorXd& variables) {
  int norbs  = Determinant::norbs;
  int nalpha = Determinant::nalpha;
  int nbeta  = Determinant::nbeta;
  int nelec =  nalpha + nbeta;

  //Use the rotation matrix variables and update the bra
  MatrixXcd U(2*norbs-nelec, nelec);
  for (int a=0; a<2*norbs-nelec; a++) 
    for (int i=0; i<nelec; i++) {
      U(a, i) = complex<double>( variables( 2* (a*nelec+i)  ),
                                 variables( 2* (a*nelec+i)+1)
                                 );
    }
  
  MatrixXcd bra = orbitals.block(0, 0, 2*norbs, nelec)
      + orbitals.block(0, nelec, 2*norbs, 2*norbs-nelec) * U;
  return bra;
}

double GetResidual::getResidue(const VectorXd& variables,
                               VectorXd& residual,
                               MatrixXd& JastrowHessian,
                               bool getJastrowResidue,
                               bool getOrbitalResidue,
                               bool getJastrowHessian) {

  int norbs  = Determinant::norbs;
  int nalpha = Determinant::nalpha;
  int nbeta  = Determinant::nbeta;
  int nelec =  nalpha + nbeta;
  
  int nJastrowVars = 2*norbs*(2*norbs+1)/2;
  int nOrbitalVars = 2*norbs*(nalpha+nbeta);
  VectorXd Jastrow = variables.block(0,0,nJastrowVars, 1);
  VectorXd U = variables.block(nJastrowVars, 0, 2*(2*norbs - nelec)*nelec, 1);
  MatrixXcd bra  = fillWfnOrbs(orbitals, U);

  //store the jastrow and orbital residue
  VectorXd JastrowResidue = 0 * Jastrow;
  VectorXd braVarsResidue( 2*(2*norbs - nelec)* nelec); braVarsResidue.setZero();

  //apply the Sz projector and generate a linear combination of kets
  vector<MatrixXcd> ket; vector<complex<double>> coeffs;
  applyProjector(bra, ket, coeffs, nalpha-nbeta, ngrid); 


  MatrixXcd S;
  double detovlp = 0.;

  //these terms are needed to calculate the orbital and jastrow gradient respectively
  MatrixXcd orbitalGradEterm = bra; orbitalGradEterm.setZero();
  VectorXd NiNjRDM(2*norbs*(2*norbs+1)/2); NiNjRDM.setZero();
  MatrixXcd mfRDM(2*norbs, 2*norbs);
  for (int g = 0; g<ket.size(); g++) {
    S = bra.adjoint()*ket[g];
    complex<double> Sdet = S.determinant();
    detovlp += (Sdet * coeffs[g]).real();
    //if (commrank == 0) cout <<g<<"  "<<ket.size()<<"  "<< Sdet<<"  "<<coeffs[g]<<endl;

    if (getOrbitalResidue)
      orbitalGradEterm += Sdet * coeffs[g] * ket[g] * S.inverse();

    if (getJastrowResidue) {
      mfRDM = ((ket[g] * S.inverse())*bra.adjoint());     

      complex<double> factor = conj(Sdet * coeffs[g]);
      auto ninjrdm = [&](const int& i, const int& j) {
        NiNjRDM(index(i, j)) +=  i == j ? (mfRDM(j, i)*factor).real()
        : ((mfRDM(j, j)*mfRDM(i, i) - mfRDM(i, j)*mfRDM(j, i))*factor).real();};

      loopOverLowerTriangle(ninjrdm);
      
    }
  }
  orbitalGradEterm /= detovlp;
  NiNjRDM /= detovlp;

  if (commrank == 0) cout << detovlp <<endl;
  
  MatrixXcd braResidueMat = 0.*bra; braResidueMat.setZero();
  double Energy = getResidueSingleKet(
      detovlp,
      bra, ket, coeffs, braResidueMat,
      Jastrow, JastrowResidue, JastrowHessian,
      getJastrowResidue, getOrbitalResidue, getJastrowHessian ) ;

  
  if (getJastrowResidue)
    JastrowResidue -= Energy *NiNjRDM;
  
  if (getOrbitalResidue) {
    braResidueMat -= Energy * orbitalGradEterm;
    
    
    MatrixXcd nonRedundantOrbResidue =
        orbitals.block(0,nelec, 2*norbs, 2*norbs - nelec).adjoint()*braResidueMat;

    //make vectors of doubles from complex matrix
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
  MatrixXd JastrowHessian;
  double Energy = getResidue(variables, residue, JastrowHessian, true, false, false);

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
  MatrixXd JastrowHessian;
  double Energy = getResidue(variables, residue, JastrowHessian, false, true, false);

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
    MatrixXd& JastrowHessian,
    bool getJastrowResidue,
    bool getOrbitalResidue,
    bool getJastrowHessian){


  size_t norbs  = Determinant::norbs;
  size_t nelec = Determinant::nalpha+Determinant::nbeta;
  calcRDM calcrdm;   
  MatrixXcd LambdaD, LambdaC, S;
  DiagonalXd diagcre(2*norbs),
      diagdes(2*norbs);  
  MatrixXcd rdm, Sinv, JJphi, JJphiSinv;
  complex<double> Energy, Sdet, factor;

  //calcualte the residual for gradient
  auto orbGrad = [&] (const int& orb1, const int& orb2, const complex<double>& f) {
    orbitalResidueMat.row(orb1) += f * (diagcre.diagonal()[orb1] * (LambdaD.row(orb2) * Sinv));
    orbitalResidueMat += f * (-((JJphiSinv*LambdaC.adjoint().col(orb1)) * (LambdaD.row(orb2) * Sinv)));
  };

  //calculate the RDMs and energy
  auto calcRDM = [&](const int& orb1, const int& orb2, const int& orb3, const int& orb4,
                     const MatrixXcd& ket, const complex<double>& coeff, const double& integral) {
    factor = getCreDesDiagMatrix(diagcre, diagdes, orb1, orb2, orb3, orb4, norbs, Jastrow);
    LambdaD = diagdes*ket;
    LambdaC = diagcre*bra;    
    S = LambdaC.adjoint()*LambdaD;    
    Sdet = S.determinant();
    Sinv = S.inverse();
    rdm = (LambdaD * Sinv) * LambdaC.adjoint();
    factor *= integral * Sdet * coeff / detovlp;
  };

  //calculate the residual for jastrow
  auto JastrowGrad = [&] (const int& orb1, const int& orb2, const int& orb3, const int& orb4) {
    auto Jres = [&](const int& orbm, const int& orbn) {
      complex<double> res = calcrdm.calcTerm1(orbm, orbn, orb1, orb2, orb3, orb4, rdm);
      JastrowResidue(index(orbm, orbn)) += (res * factor).real();
    };    
    loopOverLowerTriangle(Jres);
  };


  //calculate the jastrow hessian
  auto JastrowHess = [&] (const int& orb1, const int& orb2, const int& orb3, const int& orb4) {
    auto Jhess = [&](const int& orbm, const int& orbn, const int& orbp, const int& orbq) {
      complex<double> res = calcrdm.calcTerm2(orbm, orbn, orb1, orb2, orb3, orb4, orbp, orbq, rdm);
      JastrowHessian(index(orbm, orbn), index(orbp, orbq)) += (res * factor).real();
    };
    
    auto JhessOuter = [&Jhess](const int& orbm, const int& orbn) {
      auto Jhesspartial = [&](const int& orbp, const int & orbq) {
        Jhess(orbm, orbn, orbp, orbq);};
      loopOverLowerTriangle( Jhesspartial);
    };
    
    //nested loop over lower triangle
    loopOverLowerTriangle(JhessOuter);
  };

  
  //loop over all the kets
  for (int g = 0; g <ketvec.size(); g++) {
    MatrixXcd& ket = ketvec[g];
    complex<double> coeff = coeffvec[g];

    //perform 1e calcs
    auto run1eCode = [&] (const int& orb1, const int& orb2, const double& integral) {
      calcRDM(orb1, orb2, -1, -1, ket, coeff, integral);
      Energy += rdm(orb2, orb1) * factor;

      if (getOrbitalResidue) {
        JJphi = diagcre*LambdaD;
        JJphiSinv = JJphi*Sinv;
        orbGrad(orb1, orb2, factor);
        orbitalResidueMat += rdm(orb2, orb1) * factor * JJphiSinv;
      }

      if (getJastrowResidue) {
        JastrowGrad(orb1, orb2, -1, -1);
      }
      
    };    
    loopOver1epar(run1eCode);
    //loopOver1e(run1eCode);


    //perform 2e calcs
    auto run2eCode = [&] (const int& orb1, const int& orb2, const int& orb3,
                          const int &orb4, const double& integral) {
      calcRDM(orb1, orb2, orb3, orb4, ket, coeff, integral);
      complex<double> Econtribution = (rdm(orb4, orb1) * rdm(orb3, orb2)
                                       - rdm(orb3, orb1) * rdm(orb4, orb2)) * factor;
      Energy += Econtribution ;
      
      //orbital residue
      if (getOrbitalResidue) {
        JJphi = diagcre*LambdaD;
        JJphiSinv = JJphi*Sinv;
        
        orbGrad(orb2, orb3, factor*rdm(orb4, orb1));
        orbGrad(orb2, orb4, -factor*rdm(orb3, orb1));
        orbGrad(orb1, orb4, factor*rdm(orb3, orb2));
        orbGrad(orb1, orb3, -factor*rdm(orb4, orb2));
        
        orbitalResidueMat += Econtribution * JJphiSinv;
        
      }

      if (getJastrowResidue) {
        JastrowGrad(orb1, orb2, orb3, orb4);
      }
      
    };
    
    loopOver2epar(run2eCode);
    //loopOver2e(run2eCode);
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
};




int index(int I, int J) {
  return max(I,J)*(max(I,J)+1)/2 + min(I,J);
}

int ABAB(const int &orb) {
  const int& norbs = Determinant::norbs;
  return (orb%norbs)* 2 + orb/norbs;
};

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

  if (orb3 == -1 && orb4 == -1)
    return getCreDesDiagMatrix(diagcre, diagdes, orb1, orb2, norbs, JA);
  
  for (int j=0; j<2*norbs; j++) {
    diagcre.diagonal()[j] = exp(-2.*JA(index(orb1, j)) - 2.*JA(index(orb2, j)));
    diagdes.diagonal()[j] = exp(+2.*JA(index(orb3, j)) + 2.*JA(index(orb4, j)));
  }

  double factor = exp( 2*JA(index(orb1, orb2)) + JA(index(orb1, orb1)) + JA(index(orb2, orb2))
                       -2*JA(index(orb3, orb4)) - JA(index(orb3, orb3)) - JA(index(orb4, orb4))) ;
  return factor;
}



void fillJastrowfromWfn(MatrixXd& Jtmp, VectorXd& JA) {

  
  int norbs = Determinant::norbs;

  auto f = [&] (int i, int j) {
    JA(index(i,j)) =  log(Jtmp(ABAB(i), ABAB(j)));
    JA(index(i,j)) /=  i==j ? 1.: 2.;
  };
  
  loopOverLowerTriangle(f);
  
}

void fillWfnfromJastrow(VectorXd& JA, MatrixXd& Jtmp) {
  int norbs = Determinant::norbs;
  
  auto f = [&] (int i, int j) {
    Jtmp(ABAB(i), ABAB(j)) = i == j ? exp(JA(index(i,j))) : exp(2.*JA(index(i,j)));
  };
  
  loopOverLowerTriangle(f);
  
}
