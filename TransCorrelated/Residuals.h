#pragma once

#include "ResidualsUtils.h"

template<typename T, typename complexT>
struct GetResidual {
  using MatrixXcT = Matrix<complexT, Dynamic, Dynamic>;
  using VectorXcT = Matrix<complexT, Dynamic, 1>;
  using MatrixXT = Matrix<T, Dynamic, Dynamic>;
  using VectorXT = Matrix<T, Dynamic, 1>;
  using DiagonalXT = Eigen::DiagonalMatrix<T, Eigen::Dynamic>;
  
  MatrixXcT& orbitals;
  VectorXT& Jredundant;
  vector<pair<int,int>>& Jmap;
  int ngrid;

  GetResidual(MatrixXcT& pOrbitals, VectorXT& Jred, vector<pair<int, int>>& NonRedJMap,
              int pngrid=4) : orbitals(pOrbitals), Jredundant(Jred), ngrid(pngrid),
                              Jmap(NonRedJMap) {};

  T getLagrangian(const VectorXT& variables,
                  bool doParallel) {
    int norbs  = Determinant::norbs;
    int nalpha = Determinant::nalpha;
    int nbeta  = Determinant::nbeta;
    int nelec =  nalpha + nbeta;

    int nJastrowVars = 2*norbs*(2*norbs+1)/2 - Jredundant.size();
    int nOrbitalVars = (2*norbs-nelec)*(nalpha+nbeta);
    VectorXT JastrowOrbVars = variables.block(0,0,nJastrowVars+2*nOrbitalVars,1);
    VectorXT residual = JastrowOrbVars; residual.setZero();
    MatrixXT JastrowHessian;
    T Energy = getResidue(JastrowOrbVars, residual, JastrowHessian, true, false, false, doParallel);

    T lagrangian = Energy;
    VectorXT z = variables.block(nJastrowVars+2*nOrbitalVars, 0, nJastrowVars, 1);
    for (int i=0; i<nJastrowVars; i++)
      lagrangian += z[i]*residual[i];

    return lagrangian;
    
  }
  
  T getVariance(const VectorXT& variables,
                bool getJastrowResidue = true,
                bool getOrbitalResidue = true,
                bool doParallel = true){
    VectorXT residual = variables; residual.setZero();
    MatrixXT JastrowHessian;
    getResidue(variables, residual, JastrowHessian, getJastrowResidue,
               getOrbitalResidue, false, doParallel);
    return residual.stableNorm();
  }
                     
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
  
    VectorXT Jastrow(2*norbs*(2*norbs+1)/2);

    //collect redundant and non-redundant variables into Jastrow
    int nJastrowVars = 2*norbs*(2*norbs+1)/2 - Jredundant.size();
    int nOrbitalVars = 2*norbs*(nalpha+nbeta);
    VectorXT Jnonred = variables.block(0,0,nJastrowVars, 1);  
    JastrowFromRedundantAndNonRedundant(Jastrow, Jredundant, Jnonred, Jmap);
  
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
    complexT detovlpCmplx(0.0);
    
    //these terms are needed to calculate the orbital and jastrow gradient respectively
    MatrixXcT orbitalGradEterm = bra; orbitalGradEterm.setZero();
    VectorXT NiNjRDM(2*norbs*(2*norbs+1)/2); NiNjRDM.setZero();
    MatrixXcT mfRDM(2*norbs, 2*norbs);
    for (int g = 0; g<ket.size(); g++) {
      S = bra.adjoint()*ket[g] ;
      complexT Sdet = S.determinant();
      detovlp += (Sdet * coeffs[g]).real();

      if (getOrbitalResidue)
        orbitalGradEterm += Sdet * coeffs[g] * ket[g] * S.inverse();

      if (getJastrowResidue) {
        mfRDM = ((ket[g] * S.inverse())*bra.adjoint());     

        //complexT factor = conj(Sdet * coeffs[g]);
        complexT factor = Sdet *coeffs[g];
        auto ninjrdm = [&](const int& i, const int& j) {
          NiNjRDM(index(i, j)) +=  i == j ? (mfRDM(j, i)*factor).real()
          : ((mfRDM(j, j)*mfRDM(i, i) - mfRDM(i, j)*mfRDM(j, i))*factor).real();};

        loopOverLowerTriangle(ninjrdm);
      
      }
    }
    orbitalGradEterm /= detovlp;
    NiNjRDM /= detovlp;

 
    MatrixXcT braResidueMat = 0.*bra; braResidueMat.setZero();

    T Energy = getResidueSingleKet(
        detovlp,
        bra, ket, coeffs, braResidueMat,
        Jastrow, JastrowResidue, JastrowHessian,
        getJastrowResidue, getOrbitalResidue, getJastrowHessian, doParallel ) ;

    if (getJastrowResidue)
      JastrowResidue -= Energy *NiNjRDM;
  
    if (getOrbitalResidue) {
      braResidueMat -= Energy * orbitalGradEterm;

      //cout << braResidueMat<<endl;exit(0);
      MatrixXcT nonRedundantOrbResidue =
          orbitals.block(0,nelec, 2*norbs, 2*norbs - nelec).adjoint()*braResidueMat;

      //make vectors of doubles from complex matrix
      int nelec = bra.cols();
      for (int i=0; i<2*norbs - nelec; i++)
        for (int j=0; j<nelec; j++) {
          braVarsResidue(2 * (i * nelec + j)    ) = nonRedundantOrbResidue(i,j).real();
          braVarsResidue(2 * (i * nelec + j) + 1) = nonRedundantOrbResidue(i,j).imag();
        }
    
    
    }

    //take all residue equations and remove redudandant ones
    VectorXT ResidueRedundant(Jredundant.rows());
    VectorXT ResidueNonRed(Jnonred.rows());
    RedundantAndNonRedundantJastrow(JastrowResidue, ResidueRedundant, ResidueNonRed, Jmap);

    residual.block(0, 0, nJastrowVars, 1) = ResidueNonRed;
    residual.block(nJastrowVars, 0, braVarsResidue.size(), 1) = braVarsResidue;
    return Energy+coreE;
  }

  T getResidueSingleKet(
      T detovlp,
      MatrixXcT& bra,
      vector<MatrixXcT>& ketvec,
      vector<complexT>& coeffvec,
      MatrixXcT& orbitalResidueMat,
      VectorXT& Jastrow,
      VectorXT& JastrowResidue,
      MatrixXT& JastrowHessian,
      bool getJastrowResidue = true,
      bool getOrbitalResidue = true,
      bool getJastrowHessian = false,
      bool doParallel = true)  {


    size_t norbs  = Determinant::norbs;
    size_t nelec = Determinant::nalpha+Determinant::nbeta;
    calcRDM<T, complexT> calcrdm;   
    MatrixXcT LambdaD, LambdaC, S;
    DiagonalXT diagcre(2*norbs),
        diagdes(2*norbs);  
    MatrixXcT rdm, Sinv, JJphi, JJphiSinv;
    complexT Energy, Sdet, factor;

    //calcualte the residual for gradient
    auto orbGrad = [&] (const int& orb1, const int& orb2, const complexT& f) {
      orbitalResidueMat.row(orb1) += f * (diagcre.diagonal()[orb1] * (LambdaD.row(orb2) * Sinv));
      orbitalResidueMat += f*(-((JJphiSinv*LambdaC.adjoint().col(orb1)) * (LambdaD.row(orb2) * Sinv)));
    };

    //calculate the RDMs and energy
    auto calculateRDM = [&](const int& orb1, const int& orb2, const int& orb3, const int& orb4,
                            const MatrixXcT& ket, const complexT& coeff, const T& integral) {
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
        complexT res = calcrdm.calcTerm1(orbm, orbn, orb1, orb2, orb3, orb4, rdm);
        JastrowResidue(index(orbm, orbn)) += (res * factor).real();
      };    
      if (schd.wavefunctionType == "JastrowSlater")
        loopOverLowerTriangle(Jres);
      else if (schd.wavefunctionType == "GutzwillerSlater")
        loopOverGutzwillerJastrow(Jres);
    };


    //calculate the jastrow hessian
    auto JastrowHess = [&] (const int& orb1, const int& orb2, const int& orb3, const int& orb4) {
      auto Jhess = [&](const int& orbm, const int& orbn, const int& orbp, const int& orbq) {
        complexT res = calcrdm.calcTerm2(orbm, orbn, orb1, orb2, orb3, orb4, orbp, orbq, rdm);
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
      MatrixXcT& ket = ketvec[g];
      complexT coeff = coeffvec[g];
      //perform 1e calcs
      auto run1eCode = [&] (const int& orb1, const int& orb2, const T& integral) {
        calculateRDM(orb1, orb2, -1, -1, ket, coeff, integral);

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
      if (doParallel)
        loopOver1epar(run1eCode);
      else
        loopOver1e(run1eCode);


      //perform 2e calcs
      auto run2eCode = [&] (const int& orb1, const int& orb2, const int& orb3,
                            const int &orb4, const T& integral) {
        calculateRDM(orb1, orb2, orb3, orb4, ket, coeff, integral);
        complexT Econtribution = (rdm(orb4, orb1) * rdm(orb3, orb2)
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
    
      if (doParallel)
        loopOver2epar(run2eCode);
      else
        loopOver2e(run2eCode);

    }

#ifndef SERIAL
    if (doParallel) {
      size_t jsize = JastrowResidue.size();
      MPI_Allreduce(MPI_IN_PLACE, &JastrowResidue(0), jsize, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

      int a = orbitalResidueMat.rows(), b = orbitalResidueMat.cols();
      size_t osize = 2*a*b;
      MPI_Allreduce(MPI_IN_PLACE, &orbitalResidueMat(0,0), osize, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      
      size_t two = 2;
      MPI_Allreduce(MPI_IN_PLACE, &Energy, two, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }
#endif
    return Energy.real();
  };

  T GetResidual::getJastrowResidue(const VectorXT& jastrowvars,
                                   const VectorXT& braVars,
                                   VectorXT& jastrowResidue,
                                   bool doParallel=true) {
    int nJastrowVars = jastrowvars.size(),
        nBraVars = braVars.size();
    VectorXT variables(nJastrowVars+nBraVars);

    variables.block(0, 0, nJastrowVars, 1) = jastrowvars;
    variables.block(nJastrowVars, 0, nBraVars, 1) = braVars;

    VectorXT residue = variables;
    MatrixXT JastrowHessian;
    T Energy = getResidue(variables, residue, JastrowHessian, true, false, false, doParallel);

    jastrowResidue = residue.block(0, 0, nJastrowVars, 1);
    return Energy;
  }

  T GetResidual::getOrbitalResidue(const VectorXT& jastrowvars,
                                   const VectorXT& braVars,
                                   VectorXT& orbitalResidue,
                                   bool doParallel = true) {
    int nJastrowVars = jastrowvars.size(),
        nBraVars = braVars.size();
    VectorXT variables(nJastrowVars+nBraVars);
    variables.block(0, 0, nJastrowVars, 1) = jastrowvars;
    variables.block(nJastrowVars, 0, nBraVars, 1) = braVars;

    VectorXT residue = variables;
    MatrixXT JastrowHessian;
    T Energy = getResidue(variables, residue, JastrowHessian, false, true, false, doParallel);

    orbitalResidue = residue.block(nJastrowVars, 0, nBraVars, 1);
    return Energy;
  }
  
};





