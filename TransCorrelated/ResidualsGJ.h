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
                  const MatrixXcT& ketvec,
                  const VectorXT& Jastrow,
                  const VectorXT& Lambda,
                  VectorXT& JastrowRes,
                  MatrixXcT& BraGrad,
                  MatrixXcT& KetGrad,
                  MatrixXcT& KetdagGrad,
                  VectorXT& JasGrad,
                  VectorXT& JasGradE,
                  MatrixXd& JasLamHess) {
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
    vector<VectorXcT> Projector;
    T Sz = 1.*(nalpha-nbeta);
    applyProjector(ketvec, ket, coeffs, Projector, Sz, ngrid); 


    MatrixXcT S;
    T detovlp = 0., detovlpPrev = 0.;
    complexT detovlpCmplx(0.0), E(0,0), Eprev(0,0);
    calcRDM<T, complexT> calcrdm;   

    VectorXT NiRDM(norbs); NiRDM.setZero();
    VectorXT Nires(norbs); Nires.setZero();
    
    MatrixXcT mfRDM(2*norbs, 2*norbs);


    //DERIVATIVE WORK ARRAY (UGLY AS HELL)*****
    MatrixXcT RDMGradE(2*norbs, 2*norbs), RDMGradRes(2*norbs, 2*norbs),
        RDMGradNi(2*norbs, 2*norbs);
    MatrixXcT KETGradE(2*norbs, nelec), KETGradRes(2*norbs, nelec),
        KETGradNi(2*norbs, nelec), KETGraddetovlp(2*norbs, nelec);
    MatrixXcT KETdagGradE(2*norbs, nelec), KETdagGradRes(2*norbs, nelec),
        KETdagGradNi(2*norbs, nelec), KETdagGraddetovlp(2*norbs, nelec);
    MatrixXcT BRAGradE(2*norbs, nelec), BRAGradRes(2*norbs, nelec),
        BRAGraddetovlp(2*norbs, nelec), BRAGradNi(2*norbs, nelec);
    MatrixXcT JASGradE(2*norbs, 2*norbs), JASGradRes(2*norbs, 2*norbs);
    RDMGradE.setZero(); RDMGradRes.setZero(); RDMGradNi.setZero();
    KETGradE.setZero(); KETGradRes.setZero();
    KETGradNi.setZero(); KETGraddetovlp.setZero();
    KETdagGradE.setZero(); KETdagGradRes.setZero();
    KETdagGradNi.setZero(); KETdagGraddetovlp.setZero();
    BRAGradE.setZero(); BRAGradRes.setZero();
    BRAGraddetovlp.setZero(); BRAGradNi.setZero();
    JASGradE.setZero(); JASGradRes.setZero();

    //VectorXd JasGrad(norbs);
    JasLamHess.resize(norbs, norbs);
    MatrixXd JasHess(norbs, norbs), JasLamHessLoc(norbs, norbs);
    JasGradE.setZero(); JasHess.setZero(); JasLamHess.setZero();
    //*********
    
    for (int g = 0; g<ket.size(); g++) {
      S = bra.adjoint()*ket[g] ;
      MatrixXcT Sinv = S.inverse();
      complexT Sdet = S.determinant();
      detovlp += (Sdet * coeffs[g]).real();
      complexT factor = Sdet *coeffs[g];
      complexT nireVal = 0.0, Ni = 0.0;
      
      mfRDM = (((ket[g] * Sinv)*bra.adjoint()));           
      
      //Jastrow residue with Energy <nia nib E>
      auto nirdm = [&](const int& i) {
        int j = i + norbs;
        complexT Val = ( (mfRDM(i, i)*mfRDM(j, j)
                          - mfRDM(i, j)*mfRDM(j, i)) * factor);
        Ni += Val * Lambda[i];
        NiRDM(i) += Val.real();
        
        RDMGradNi(i, i) += factor * mfRDM(j, j) * Lambda[i];
        RDMGradNi(j, j) += factor * mfRDM(i, i) * Lambda[i];
        RDMGradNi(i, j) -= factor * mfRDM(j, i) * Lambda[i];
        RDMGradNi(j, i) -= factor * mfRDM(i, j) * Lambda[i];
      };
      loopOverAlphaOrbs(nirdm);      

      auto oneRDM = [&](const int& i, const int &j, const double& i1) {
        int I = getOtherSpin(i, norbs);
        int J = getOtherSpin(j, norbs);
        T xiI = ximinus[min(i,I)], xiJ = xiplus[min(j,J)];
        E += (calcrdm.calcTermGJ1(i, j, mfRDM, I, J, xiI, xiJ,
                                  RDMGradE, factor*i1, JasGradE,
                                  JasHess));

        auto ni1e = [&](const int& orb1) {
          int orb1b = orb1+norbs;
          complexT Val = (calcrdm.calcTermGJ1jas(orb1, orb1b, i, j, mfRDM,
                                                 I, J, xiI, xiJ,
                                                 RDMGradRes, factor*i1,
                                                 Lambda[orb1], JasLamHess));
          Nires[orb1] += Val.real();
          nireVal += Val * Lambda[orb1];
          
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
        T xiI = ximinus[min(i,I)], xiJ = ximinus[min(j,J)],
        xiK = xiplus[min(k,K)], xiL = xiplus[min(l,L)];

        T f1 = abs(i - j ) == norbs ? exp(2*Jastrow(min(i,j))) : 1.0;
        f1 *= abs(k - l) == norbs ? exp(-2*Jastrow(min(k,l))) : 1.0;

        auto Eloc = calcrdm.calcTermGJ2(i, j, k, l, mfRDM, I, J, K, L, xiI, xiJ, xiK, xiL,
                                         RDMGradE, f1 * factor * i2, JasGradE, JasHess);

        if (abs(i-j) == norbs) JasGradE[min(i,j)] += 2. * Eloc.real();
        if (abs(k-l) == norbs) JasGradE[min(k,l)] -= 2. * Eloc.real();
        E += Eloc;

        auto ni2e = [&](const int& orb1) {
          int orb1b = orb1+norbs;
          complexT Val = calcrdm.calcTermGJ2jas(orb1, orb1b, i, j, k, l,
                                                mfRDM, I, J, K, L,
                                                xiI, xiJ, xiK, xiL,
                                                RDMGradRes, f1*factor*i2,
                                                Lambda[orb1], JasLamHess);
          
          if (abs(i-j) == norbs) JasLamHess(orb1, min(i,j)) += 2. * Val.real();
          if (abs(k-l) == norbs) JasLamHess(orb1, min(k,l)) -= 2. * Val.real();
          
          Nires[orb1] += Val.real();
          nireVal += Val * Lambda[orb1];
        };
        loopOverAlphaOrbs(ni2e);
        
      };
      loopOver2epar(twoRDM);


      //Derivative w.r.t. BRA
      {      
        BRAGradE += RDMGradE.transpose() * ket[g] * Sinv;
        BRAGradE -= mfRDM * RDMGradE.transpose() * ket[g] * Sinv;
        BRAGradE +=  (ket[g]*Sinv) * (E - Eprev);
        
        BRAGradRes += RDMGradRes.transpose() * ket[g] * Sinv;
        BRAGradRes -= mfRDM * RDMGradRes.transpose() * ket[g] * Sinv;
        BRAGradRes +=  (ket[g]*Sinv) * (nireVal);
        
        BRAGradNi += RDMGradNi.transpose() * ket[g] * Sinv;
        BRAGradNi -= mfRDM * RDMGradNi.transpose() * ket[g] * Sinv;
        BRAGradNi +=  (ket[g]*Sinv) * (Ni);
        
        BRAGraddetovlp +=  (ket[g]*Sinv) * Sdet * coeffs[g];
      }

      //Derivative w.r.t. KET
      {
        auto SB = Sinv * bra.adjoint();
        auto diagMat = Projector[g].asDiagonal();
        auto diagMatadj = Projector[g].conjugate().asDiagonal();
        if (g%2 == 0) {
          KETGradE += diagMat*(RDMGradE* SB.transpose() );
          KETGradE -= diagMat*(mfRDM.transpose() * RDMGradE * SB.transpose());
          KETGradE += diagMat*(bra.conjugate() * Sinv.transpose()) * (E - Eprev);
          
          KETGradRes += diagMat*(RDMGradRes* SB.transpose() );
          KETGradRes -= diagMat*(mfRDM.transpose() * RDMGradRes * SB.transpose());
          KETGradRes += diagMat*(bra.conjugate() * Sinv.transpose()) * (nireVal);
          
          KETGradNi += diagMat*(RDMGradNi* SB.transpose() );
          KETGradNi -= diagMat*(mfRDM.transpose() * RDMGradNi * SB.transpose());
          KETGradNi += diagMat*(bra.conjugate() * Sinv.transpose()) * (Ni);
        
          KETGraddetovlp +=  diagMat*(bra.conjugate() * Sinv.transpose()) * Sdet * coeffs[g];
        }
        else {
          KETdagGradE += diagMatadj*(RDMGradE* SB.transpose() );
          KETdagGradE -= diagMatadj*(mfRDM.transpose() * RDMGradE * SB.transpose());
          KETdagGradE += diagMatadj*(bra.conjugate() * Sinv.transpose()) * (E - Eprev);
          
          KETdagGradRes += diagMatadj*(RDMGradRes* SB.transpose() );
          KETdagGradRes -= diagMatadj*(mfRDM.transpose() * RDMGradRes * SB.transpose());
          KETdagGradRes += diagMatadj*(bra.conjugate() * Sinv.transpose()) * (nireVal);
          
          KETdagGradNi += diagMatadj*(RDMGradNi* SB.transpose() );
          KETdagGradNi -= diagMatadj*(mfRDM.transpose() * RDMGradNi * SB.transpose());
          KETdagGradNi += diagMatadj*(bra.conjugate() * Sinv.transpose()) * (Ni);
        
          KETdagGraddetovlp +=  diagMatadj*(bra.conjugate() * Sinv.transpose()) * Sdet * coeffs[g];

        }
      }

      //cout << g<<"  "<<KETGradE(0,0)<<"  "<<KETdagGradE(0,0)<<"  "<<BRAGradE(0,0)<<endl;
      //complexT out = E + Lambda.dot(Nires);
      //return (out).real();
      
      Eprev = E;
      nireVal = 0.0;
      Ni = 0.0;
      
      RDMGradE.setZero();
      RDMGradRes.setZero();
      RDMGradNi.setZero();
    }

#ifndef SERIAL
    size_t jsize = Nires.size();
    MPI_Allreduce(MPI_IN_PLACE, &Nires(0), jsize, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    size_t two = 2;
    MPI_Allreduce(MPI_IN_PLACE, &E, two, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

    JastrowRes = (Nires - E.real()/detovlp * NiRDM);
    T Lagrangian = E.real() + Lambda.dot(JastrowRes);

    BraGrad = (BRAGradE * (1. - Lambda.dot(NiRDM)/detovlp)
               + BRAGradRes - E.real()/detovlp * BRAGradNi
               + E.real()/detovlp/detovlp * Lambda.dot(NiRDM)*BRAGraddetovlp)/detovlp
        - Lagrangian/detovlp/detovlp * BRAGraddetovlp;

    KetGrad = (KETGradE * (1. - Lambda.dot(NiRDM)/detovlp)
               + KETGradRes - E.real()/detovlp * KETGradNi
               + E.real()/detovlp/detovlp * Lambda.dot(NiRDM)*KETGraddetovlp)/detovlp
        - Lagrangian/detovlp/detovlp * KETGraddetovlp;

    KetdagGrad = (KETdagGradE * (1. - Lambda.dot(NiRDM)/detovlp)
                  + KETdagGradRes - E.real()/detovlp * KETdagGradNi
                  + E.real()/detovlp/detovlp * Lambda.dot(NiRDM)*KETdagGraddetovlp)/detovlp
        - Lagrangian/detovlp/detovlp * KETdagGraddetovlp;



    JasGrad = JasGradE*(1.0 - Lambda.dot(NiRDM)/detovlp);
    JasGrad += (Lambda.transpose()*JasLamHess).transpose();
    JasGrad /= detovlp;

    for (int l=0; l<norbs; l++)
      for (int r=0; r<norbs; r++)
        JasLamHess(l, r) -= NiRDM(l) * JasGradE(r) / detovlp;

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
    vector<VectorXcT> Projector;
    applyProjector(bra, ket, coeffs, Projector, Sz, ngrid); 

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





