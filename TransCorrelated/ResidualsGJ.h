#pragma once

#include "ResidualsUtils.h"
#include "calcRDM.h"

using namespace std;
using namespace Eigen;

template<typename T, typename complexT>
struct GetResidualGJ {
  using MatrixXcT = Matrix<complexT, Dynamic, Dynamic>;
  using VectorXcT = Matrix<complexT, Dynamic, 1>;
  using MatrixXT = Matrix<T, Dynamic, Dynamic>;
  using VectorXT = Matrix<T, Dynamic, 1>;
  using DiagonalXT = Eigen::DiagonalMatrix<T, Eigen::Dynamic>;
  
  int ngrid;

  GetResidualGJ(int pngrid=4) : ngrid(pngrid)
  {};

  int getOtherSpin(int I, int norbs) {
    return I >= norbs ? I - norbs : I + norbs;
  }

  T getLagrangian(const MatrixXcT& bra,
                  const VectorXT& Jastrow,
                  const VectorXT& Lambda) {
    int norbs  = Determinant::norbs;
    int nalpha = Determinant::nalpha;
    int nbeta  = Determinant::nbeta;
    int nelec =  nalpha + nbeta;

    VectorXT xiplus(norbs), ximinus(norbs);
    for (int i=0; i<norbs; i++) {
      xiplus[i]  = exp( 2.*Jastrow[i]) - 1;
      ximinus[i] = exp(-2.*Jastrow[i]) - 1;
    }

    //apply the Sz projector and generate a linear combination of kets
    vector<MatrixXcT> ket(2*ngrid); vector<complexT> coeffs(2*ngrid);
    T Sz = 1.*(nalpha-nbeta);
    applyProjector(bra, ket, coeffs, Sz, ngrid); 


    MatrixXcT S;
    T detovlp = 0.;
    complexT detovlpCmplx(0.0), E(0,0);
    calcRDM<complexT> calcrdm;   

    VectorXT NiRDM(norbs); NiRDM.setZero();
    VectorXT Nires(norbs); Nires.setZero();
    
    MatrixXcT mfRDM(2*norbs, 2*norbs), oneRDMExp(2*norbs, 2*norbs);
    for (int g = 0; g<ket.size(); g++) {
      S = bra.adjoint()*ket[g] ;
      complexT Sdet = S.determinant();
      detovlp += (Sdet * coeffs[g]).real();
      complexT factor = Sdet *coeffs[g];

      mfRDM = ((ket[g] * S.inverse())*bra.adjoint());     

      //Jastrow residue with Energy <nia nib E>
      auto nirdm = [&](const int& i) {
        int j = i + norbs;
        NiRDM(i) +=
        ((mfRDM(j, j)*mfRDM(i, i) - mfRDM(i, j)*mfRDM(j, i))*factor).real();
      };
      loopOverAlphaOrbs(nirdm);      

      //<H> for hubbard
      //onerdm expectation
      oneRDMExp = mfRDM;
      auto oneRDM = [&](const int& i, const int &j, const double& i1) {
        int I = getOtherSpin(i, norbs);
        int J = getOtherSpin(j, norbs);
        double xiI = ximinus[min(i,I)], xiJ = xiplus[min(j,J)];
        E += factor * i1 * (calcrdm.calcTermGJ1(i, j, mfRDM, I, J, xiI, xiJ) + mfRDM(j,i));

        auto ni1e = [&](const int& orb1) {
          int orb1b = orb1+norbs;
          Nires[orb1] += (factor * i1 * calcrdm.calcTermGJ1jas(orb1, orb1b, i, j,
                                                               mfRDM, I, J, xiI, xiJ)).real();
        };
        loopOverAlphaOrbs(ni1e);
      };
      loopOver1epar(oneRDM);
      

      
      auto twoRDM = [&](const int& i, const int& j, const int& k,
                        const int& l, const double& i2) {

        int I = getOtherSpin(i, norbs);
        int J = getOtherSpin(j, norbs);
        int K = getOtherSpin(k, norbs);
        int L = getOtherSpin(l, norbs);
        double xiI = ximinus[min(i,I)], xiJ = ximinus[min(j,J)],
        xiK = xiplus[min(k,K)], xiL = xiplus[min(l,L)];

        double f1 = abs(i - j ) == norbs ? exp(2*Jastrow(min(i,j))) : 1.0;
        f1 *= abs(k - l) == norbs ? exp(-2*Jastrow(min(k,l))) : 1.0;

        E += f1 * factor * i2 * calcrdm.calcTermGJ2(i, j, k, l, mfRDM, I, J, K, L, xiI, xiJ, xiK, xiL);
        auto ni2e = [&](const int& orb1) {
          int orb1b = orb1+norbs;
          Nires[orb1] += (f1 * factor * i2 *
                          calcrdm.calcTermGJ2jas(orb1, orb1b, i, j, k, l, mfRDM,
                                                 I, J, K, L, xiI, xiJ, xiK, xiL)).real();
        };
        loopOverAlphaOrbs(ni2e);
        
      };
      loopOver2epar(twoRDM);
      
    }

    VectorXT JastrowRes(norbs);
    JastrowRes = (Nires - E.real()/detovlp * NiRDM)/detovlp;
    cout << JastrowRes<<endl<<endl;
    T Lagrangian = E.real() + Lambda.dot(JastrowRes);
    return (Lagrangian/detovlp);
  };

  
  //get both orbital and jastrow residues
  T getResidue(const VectorXT& variables,
               VectorXT& residual,
               MatrixXT& JastrowHessian,
               bool getJastrowResidue = true,
               bool getOrbitalResidue = true,
               bool getJastrowHessian = false,
               bool doParallel = true) {

    int norbs  = Determinant::norbs;
    int nalpha = Determinant::nalpha;
    int nbeta  = Determinant::nbeta;
    int nelec =  nalpha + nbeta;
  
    int nJastrowVars = norbs;
    int nOrbitalVars = 2*norbs*(nalpha+nbeta);
    VectorXT Jastrow = variables.block(0,0,nJastrowVars, 1);  
    VectorXT xiplus(norbs), ximinus(norbs);
    for (int i=0; i<norbs; i++) {
      xiplus[i]  = exp( 2.*Jastrow[i]) - 1;
      ximinus[i] = exp(-2.*Jastrow[i]) - 1;
    }
  
    VectorXT U = variables.block(nJastrowVars, 0, 2*(2*norbs - nelec)*nelec, 1);
    MatrixXcT&& bra  = fillWfnOrbs(orbitals, U);

    //store the jastrow and orbital residue
    VectorXT JastrowResidue = 0 * Jastrow;
    VectorXT braVarsResidue( 2*(2*norbs - nelec)* nelec); braVarsResidue.setZero();
    
    //apply the Sz projector and generate a linear combination of kets
    vector<MatrixXcT> ket(2*ngrid); vector<complexT> coeffs(2*ngrid);
    T Sz = 1.*(nalpha-nbeta);
    applyProjector(bra, ket, coeffs, Sz, ngrid); 

    MatrixXcT S;
    T detovlp = 0.;
    complexT detovlpCmplx(0.0), E(0,0);
    calcRDM<complexT> calcrdm;   
    
    //these terms are needed to calculate the orbital and jastrow gradient respectively
    VectorXT NiRDM(norbs); NiRDM.setZero();
    MatrixXcT mfRDM(2*norbs, 2*norbs), oneRDMExp(2*norbs, 2*norbs);
    for (int g = 0; g<ket.size(); g++) {
      S = bra.adjoint()*ket[g] ;
      complexT Sdet = S.determinant();
      detovlp += (Sdet * coeffs[g]).real();
      complexT factor = Sdet *coeffs[g];

      mfRDM = ((ket[g] * S.inverse())*bra.adjoint());     

      //Jastrow residue with Energy <nia nib E>
      if (getJastrowResidue) {
        //complexT factor = conj(Sdet * coeffs[g]);
        auto nirdm = [&](const int& i) {
          int j = i + norbs;
          NiRDM(i) +=
          ((mfRDM(j, j)*mfRDM(i, i) - mfRDM(i, j)*mfRDM(j, i))*factor).real();
        };
        loopOverAlphaOrbs(nirdm);      
      }


      //<H> for hubbard
      //onerdm expectation
      oneRDMExp = mfRDM;
      auto oneRDM = [&](const int& i, const int &j, const double& integral) {
        int I = getOtherSpin(i, norbs);
        int J = getOtherSpin(j, norbs);
        double xiI = ximinus[min(i,I)], xiJ = xiplus[min(j,J)];
        E += factor * integral * (calcrdm.calcTermGJ1(i, j, mfRDM, I, J, xiI, xiJ) + mfRDM(j,i));
      };
      loopOver1epar(oneRDM);
      
      //hubbard model only
      auto twoRDM = [&](const int& i, const int& j, const int& k,
                        const int& l, const double& i2) {
        E += factor * i2 * (mfRDM(l,i)*mfRDM(k,j) - mfRDM(k,i)*mfRDM(l,j));
      };
      loopOver2epar(twoRDM);
      
    }

    NiRDM /= detovlp;
    
    
    return (E/detovlp).real();
  };
  
  
};





