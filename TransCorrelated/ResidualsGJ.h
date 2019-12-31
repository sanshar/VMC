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
  MatrixXcT U2, V2;
  Matrix<complexT, 2, 2> C2, work2, C2inv;

  MatrixXcT U4, V4, ASU4, V4SB, rdmintermediate;
  Matrix<complexT, 4, 4> C4, work4, C4inv;
  
  GetResidualGJ(int pngrid=4) : ngrid(pngrid)
  {};


  int getOtherSpin(int I, int norbs) {
    return I >= norbs ? I - norbs : I + norbs;
  }

  void braVarsToMatrix(const VectorXT& bravars,
                       MatrixXcT& bra) {
    int norbs  = Determinant::norbs;
    int nalpha = Determinant::nalpha;
    int nbeta  = Determinant::nbeta;
    int nelec =  nalpha + nbeta;
    int index = 0;
    for (int i=0; i<2*norbs; i++)
      for (int j=0; j<nelec; j++) {
        bra(i,j).real(bravars[index]); index++;
        bra(i,j).imag(bravars[index]); index++;
      }
  }

  T getOrbitalGradient(const VectorXT& bravars,
                       const VectorXT& Jastrow,
             const VectorXT& Lambda) {
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
    calcRDM<T, complexT> calcrdm;   

    VectorXT NiRDM(norbs); NiRDM.setZero();
    VectorXT Nires(norbs); Nires.setZero();
    VectorXT DiagCre(2*norbs), DiagDes(2*norbs);
    MatrixXcT U(nelec, 4), V(4, nelec);
    
    MatrixXcT mfRDM(2*norbs, 2*norbs), oneRDM2e(2*norbs, 2*norbs),
        Sinv(2*norbs, 2*norbs), Slocinv(2*norbs, 2*norbs);
    MatrixXcT LambdaD(2*norbs, nelec), LambdaC(2*norbs, nelec);

    Eigen::Matrix<complexT, 4, 4> VU, I2;  I2.setIdentity();
    Eigen::Matrix<complexT, 4, 1> C;

    
    for (int g = 0; g<ket.size(); g++) {
      S = bra.adjoint()*ket[g] ;
      complexT Sdet = S.determinant();
      detovlp += (Sdet * coeffs[g]).real();
      complexT factor = Sdet *coeffs[g];
      MatrixXcT Sinv = S.inverse();
      
      mfRDM = ((ket[g] * Sinv)*bra.adjoint());     

      //Jastrow residue with Energy <nia nib E>
      auto nirdm = [&](const int& i) {
        int j = i + norbs;
        NiRDM(i) +=
        ((mfRDM(j, j)*mfRDM(i, i) - mfRDM(i, j)*mfRDM(j, i))*factor).real();
      };
      loopOverAlphaOrbs(nirdm);      

      MatrixXcT ketSinv = ket[g]*Sinv, Sinvbra=Sinv*bra.adjoint();
      MatrixXcT ketSinvloc = ket[g]*Sinv, Sinvbraloc=Sinv*bra.adjoint();
      
      //onerdm expectation
      auto oneRDM = [&](const int& i, const int &j, const double& i1) {
        int I = getOtherSpin(i, norbs);
        int J = getOtherSpin(j, norbs);
        T factorI = exp(-2.*Jastrow[min(i,I)]),
        factorJ = exp( 2.*Jastrow[min(j,J)]);

        oneRDM2e = mfRDM;
        oneRDM2e.row(I) = factorI * mfRDM.row(I);
        oneRDM2e.col(J) = factorJ * mfRDM.col(J);
        oneRDM2e(I,J) = (factorI * factorJ) * mfRDM(I, J);

        complexT detFactor(1.0, 0.0);
        if (i != j) {
          ketSinvloc.row(J) *= factorJ; 
          Sinvbraloc.col(I) *= factorI;
          detFactor = OneelectronUpdateRDM(ketSinvloc, Sinvbraloc, Jastrow, oneRDM2e, i, j, Sinv, bra, ket[g]);
          ketSinvloc.row(J) /= factorJ; 
          Sinvbraloc.col(I) /= factorI;
        }
        
        E += factor * i1 * (oneRDM2e(j,i)) * detFactor;

        auto ni1e = [&](const int& orb1) {
          int orb1b = orb1+norbs;
          Nires[orb1] += (factor * i1 * calcrdm.calcTerm1(orb1, orb1b, i, j, -1, -1, oneRDM2e) * detFactor).real();
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

        T factorI = exp(-2.*Jastrow[min(i,I)]),
        factorJ = exp(-2.*Jastrow[min(j,J)]),
        factorK = exp( 2.*Jastrow[min(k,K)]),
        factorL = exp( 2.*Jastrow[min(l,L)]);
        
        oneRDM2e = mfRDM;
        oneRDM2e.row(K) = (factorK) * mfRDM.row(K);
        oneRDM2e.row(L) = (factorL) * mfRDM.row(L);
        oneRDM2e.col(I) = (factorI) * mfRDM.col(I);
        oneRDM2e.col(J) = (factorJ) * mfRDM.col(J);
        oneRDM2e(K,I) = (factorI*factorK) * mfRDM(K,I);
        oneRDM2e(K,J) = (factorJ*factorK) * mfRDM(K,J);
        oneRDM2e(L,I) = (factorI*factorL) * mfRDM(L,I);
        oneRDM2e(L,J) = (factorJ*factorL) * mfRDM(L,J);

        ketSinvloc.row(K) *= factorK; ketSinvloc.row(L) *= factorL;
        Sinvbraloc.col(I) *= factorI; Sinvbraloc.col(J) *= factorJ;
        complexT detFactor = efficientUpdateRDM(ketSinvloc, Sinvbraloc, Jastrow, oneRDM2e,
                                                 i, j, k, l, Sinv, bra, ket[g]);
        ketSinvloc.row(K) /= factorK; ketSinvloc.row(L) /= factorL;
        Sinvbraloc.col(I) /= factorI; Sinvbraloc.col(J) /= factorJ;
        
        T f1 = abs(i - j ) == norbs ? exp(2*Jastrow(min(i,j))) : 1.0;
        f1 *= abs(k - l) == norbs ? exp(-2*Jastrow(min(k,l))) : 1.0;
        
        
        E += f1 * factor * i2 * (oneRDM2e(l,i)*oneRDM2e(k,j) - oneRDM2e(l,j)*oneRDM2e(k,i)) * detFactor;
        auto ni2e = [&](const int& orb1) {
          int orb1b = orb1+norbs;
          Nires[orb1] += (f1 * factor * i2 * calcrdm.calcTerm1(orb1, orb1b, i, j, k, l, oneRDM2e) * detFactor).real();
        };
        loopOverAlphaOrbs(ni2e);
        
      };
      //loopOver2e(twoRDM);
      loopOver2epar(twoRDM);
      
    }

    VectorXT JastrowRes(norbs);
    JastrowRes = (Nires - E.real()/detovlp * NiRDM)/detovlp;

#ifndef SERIAL
    size_t jsize = JastrowRes.size() * sizeof(JastrowRes[0])/sizeof(double);
    MPI_Allreduce(MPI_IN_PLACE, &JastrowRes(0), jsize, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    size_t two = sizeof(E);
    MPI_Allreduce(MPI_IN_PLACE, &E, two, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

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
    calcRDM<T, complexT> calcrdm;   
    
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
        T xiI = ximinus[min(i,I)], xiJ = xiplus[min(j,J)];
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
  

  complexT OneelectronExcite(int m, int n,
                         double& Mfactor, double& Nfactor,
                         const MatrixXcT& S, MatrixXcT &Sloc,
                         const MatrixXcT& bra, const MatrixXcT& ket) {
    int norbs  = Determinant::norbs;
    int nalpha = Determinant::nalpha;
    int nbeta  = Determinant::nbeta;
    int nelec =  nalpha + nbeta;
    if (U2.rows() != nelec) {
      U2.resize(nelec, 2); V2.resize(2, nelec);
      C2.setZero(); C2inv.setZero();
    }
    
    int M = getOtherSpin(m, norbs);
    int N = getOtherSpin(n, norbs);
    U2.col(0) = bra.row(M).adjoint();
    U2.col(1) = bra.row(N).adjoint();
    V2.row(0) = ket.row(M);
    V2.row(1) = ket.row(N);

    C2(0,0)    = Mfactor;    C2(1,1)    = Nfactor;
    C2inv(0,0) = 1./Mfactor; C2inv(1,1) = 1./Nfactor;
    //Sloc = S + U2*C2.asDiagonal()*V2;
    work2 = C2inv + (V2*S)*U2;
    Sloc = S - (S * U2) * (work2.inverse()) * (V2 * S);
    return work2.determinant()*C2.determinant();

  }

  complexT TwoelectronExcite(int i, int j, int k, int l, 
                         double& Ifactor, double& Jfactor,
                         double& Kfactor, double& Lfactor,
                         const MatrixXcT& S, MatrixXcT &Sloc,
                         const MatrixXcT& bra, const MatrixXcT& ket) {
    int norbs  = Determinant::norbs;
    int nalpha = Determinant::nalpha;
    int nbeta  = Determinant::nbeta;
    int nelec =  nalpha + nbeta;
    if (U4.rows() != nelec) {
      U4.resize(nelec, 4); V4.resize(4, nelec);
      C4.setZero(); C4inv.setZero(); 
    }
    
    int I = getOtherSpin(i, norbs);
    int J = getOtherSpin(j, norbs);
    int K = getOtherSpin(k, norbs);
    int L = getOtherSpin(l, norbs);
    U4.col(0) = bra.row(I).adjoint();
    U4.col(1) = bra.row(J).adjoint();
    U4.col(2) = bra.row(K).adjoint();
    U4.col(3) = bra.row(L).adjoint();
    V4.row(0) = ket.row(I);
    V4.row(1) = ket.row(J);
    V4.row(2) = ket.row(K);
    V4.row(3) = ket.row(L);

    C4(0,0) = Ifactor; C4(1,1) = Jfactor;
    C4(2,2) = Kfactor; C4(3,3) = Lfactor;
    C4inv(0,0) = 1./Ifactor; C4inv(1,1) = 1./Jfactor;
    C4inv(2,2) = 1./Kfactor; C4inv(3,3) = 1./Lfactor;
    
    //Sloc = S + U4*C4.asDiagonal()*V4;  //Sherman-Morrison Formula
    work4 = (C4inv + (V4*S)*U4);
    Sloc = S - (S * U4) * (work4.inverse()) * (V4 * S);
    return work4.determinant()*C4.determinant();
  }
  
  complexT efficientUpdate(const MatrixXcT& S, const VectorXT& Jastrow,
                       int i, int j, int k, int l,
                       MatrixXcT& Sloc, const MatrixXcT& bra,
                       const MatrixXcT& ket) {
    int norbs  = Determinant::norbs;
    int I = getOtherSpin(i, norbs);
    int J = getOtherSpin(j, norbs);
    int K = getOtherSpin(k, norbs);
    int L = getOtherSpin(l, norbs);
    
    if (i == k && j == l) //no update needed
      Sloc = S;
    else if (i == l && j == k) //no update needed
      Sloc = S;
    else if (i == l) {
      double Mfactor =  exp(-2*Jastrow[min(j,J)]) - 1;
      double Nfactor =  exp( 2*Jastrow[min(k,K)]) - 1;
      return OneelectronExcite(j, k, Mfactor, Nfactor, S, Sloc, bra, ket);
    }
    else if (i == k) {
      double Mfactor =  exp(-2*Jastrow[min(j,J)]) - 1;
      double Nfactor =  exp( 2*Jastrow[min(l,L)]) - 1;
      return OneelectronExcite(j, l, Mfactor, Nfactor, S, Sloc, bra, ket);      
    }
    else if (j == l) {
      double Mfactor =  exp(-2*Jastrow[min(i,I)]) - 1;
      double Nfactor =  exp( 2*Jastrow[min(k,K)]) - 1;
      return OneelectronExcite(i, k, Mfactor, Nfactor, S, Sloc, bra, ket);      
    }
    else if (j == k) {
      double Mfactor =  exp(-2*Jastrow[min(i,I)]) - 1;
      double Nfactor =  exp( 2*Jastrow[min(l,L)]) - 1;
      return OneelectronExcite(i, l, Mfactor, Nfactor, S, Sloc, bra, ket);      
    }
    else {
      double Ifactor = exp(-2*Jastrow[min(i,I)]) - 1,
          Jfactor = exp(-2*Jastrow[min(j,J)]) - 1,
          Kfactor = exp( 2*Jastrow[min(k,K)]) - 1,
          Lfactor = exp( 2*Jastrow[min(l,L)]) - 1;
      return TwoelectronExcite(i, j, k, l, Ifactor, Jfactor, Kfactor, Lfactor,
                        S, Sloc, bra, ket);
    }
    
    return 1.0;
  }


  complexT OneelectronUpdateRDM(const MatrixXcT& AS, const MatrixXcT& SB,
                                const VectorXT& Jastrow, MatrixXcT& rdm,
                                int m, int n,
                                const MatrixXcT& S,
                                const MatrixXcT& bra,
                                const MatrixXcT& ket) {
    int norbs  = Determinant::norbs;
    int nalpha = Determinant::nalpha;
    int nbeta  = Determinant::nbeta;
    int nelec =  nalpha + nbeta;
    if (U2.rows() != nelec) {
      U2.resize(nelec, 2); V2.resize(2, nelec);
      C2.setZero(); C2inv.setZero();
    }
    
    int M = getOtherSpin(m, norbs);
    int N = getOtherSpin(n, norbs);
    U2.col(0) = bra.row(M).adjoint();
    U2.col(1) = bra.row(N).adjoint();
    V2.row(0) = ket.row(M);
    V2.row(1) = ket.row(N);

    C2(0,0)    = exp(-2.*Jastrow[min(m,M)])-1;    C2(1,1)    = exp(2.*Jastrow[min(n,N)])-1;
    C2inv(0,0) = 1./C2(0,0); C2inv(1,1) = 1./C2(1,1);

    work2 = C2inv + (V2*S)*U2;
    //Sloc = S - (S * U2) * (work2.inverse()) * (V2 * S);
    rdm -= ((AS*U2) * work2.inverse())*(V2*SB);
    return work2.determinant()*C2.determinant();

  }

  complexT TwoelectronUpdateRDM(const MatrixXcT& AS, const MatrixXcT& SB,
                                const VectorXT& Jastrow, MatrixXcT& rdm,
                                int i, int j, int k, int l,
                                const MatrixXcT& S,
                                const MatrixXcT& bra,
                                const MatrixXcT& ket) {
    int norbs  = Determinant::norbs;
    int nalpha = Determinant::nalpha;
    int nbeta  = Determinant::nbeta;
    int nelec =  nalpha + nbeta;
    if (U4.rows() != nelec) {
      U4.resize(nelec, 4); V4.resize(4, nelec);
      C4.setZero(); C4inv.setZero();
      rdmintermediate.resize(2*norbs, 2*norbs);
      ASU4.resize(2*norbs, 4); V4SB.resize(4, 2*norbs);
    }
    
    int I = getOtherSpin(i, norbs);
    int J = getOtherSpin(j, norbs);
    int K = getOtherSpin(k, norbs);
    int L = getOtherSpin(l, norbs);
    U4.col(0) = bra.row(I).adjoint();
    U4.col(1) = bra.row(J).adjoint();
    U4.col(2) = bra.row(K).adjoint();
    U4.col(3) = bra.row(L).adjoint();
    V4.row(0) = ket.row(I);
    V4.row(1) = ket.row(J);
    V4.row(2) = ket.row(K);
    V4.row(3) = ket.row(L);

    C4(0,0) = exp(-2*Jastrow[min(i,I)])-1; C4(1,1) = exp(-2*Jastrow[min(j,J)])-1;
    C4(2,2) = exp( 2*Jastrow[min(k,K)])-1; C4(3,3) = exp( 2*Jastrow[min(l,L)])-1;
    C4inv(0,0) = 1./C4(0,0); C4inv(1,1) = 1./C4(1,1);
    C4inv(2,2) = 1./C4(2,2); C4inv(3,3) = 1./C4(3,3);
    
    //Sloc = S + U4*C4.asDiagonal()*V4;  //Sherman-Morrison Formula
    work4 = (C4inv + (V4*S)*U4);
    //Sloc = S - (S * U4) * (work4.inverse()) * (V4 * S);
    ASU4 = AS*U4; V4SB = V4*SB;
    rdmintermediate = (ASU4 * work4.inverse()) * V4SB;
    rdm -= rdmintermediate;
    return work4.determinant()*C4.determinant();
  }
  
  complexT efficientUpdateRDM(const MatrixXcT& AS, const MatrixXcT& SB,
                              const VectorXT& Jastrow, MatrixXcT& rdm,
                              int i, int j, int k, int l,
                              const MatrixXcT& S,
                              const MatrixXcT& bra,
                              const MatrixXcT& ket) {
    T retval = 1.0;
    if (i == k && j == l) //no update needed
      return retval;
    else if (i == l && j == k) //no update needed
      return retval;
    else if (i == l) {
      return OneelectronUpdateRDM(AS, SB, Jastrow, rdm, j, k, S, bra, ket);
    }
    else if (i == k) {
      return OneelectronUpdateRDM(AS, SB, Jastrow, rdm, j, l, S, bra, ket);
      //return OneelectronUpdateRDM(j, l, Mfactor, Nfactor, S, Sloc, bra, ket);      
    }
    else if (j == l) {
      return OneelectronUpdateRDM(AS, SB, Jastrow, rdm, i, k, S, bra, ket);
      //return OneelectronUpdateRDM(i, k, Mfactor, Nfactor, S, Sloc, bra, ket);      
    }
    else if (j == k) {
      return OneelectronUpdateRDM(AS, SB, Jastrow, rdm, i, l, S, bra, ket);
      //return OneelectronUpdateRDM(i, l, Mfactor, Nfactor, S, Sloc, bra, ket);      
    }
    else {
      return TwoelectronUpdateRDM(AS, SB, Jastrow, rdm, i, j, k, l, S, bra, ket);
      //return TwoelectronUpdateRDM(i, j, k, l, Ifactor, Jfactor, Kfactor, Lfactor,
      //S, Sloc, bra, ket);
    }
    
  }
  
};





