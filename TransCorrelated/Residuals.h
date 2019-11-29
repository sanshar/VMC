#pragma once

#include <Eigen/Dense>
#include <vector>
#include "Complex.h"
#include "Determinants.h"
#include "calcRDM.h"

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


void ConstructRedundantJastrowMap(vector<pair<int,int>>& NonRedundantMap) ;

template<typename T>
void RedundantAndNonRedundantJastrow(Matrix<T, Dynamic, 1>& JA, Matrix<T, Dynamic, 1>& Jred,
                                     Matrix<T, Dynamic, 1>& JnonRed,
                                     vector<pair<int, int>>& NonRedundantMap) {

  int norbs = Determinant::norbs;

  //diagonal jastrow are redundant
  for (int i=0; i<2*norbs; i++)
    Jred(i) = JA(index(i,i));
  for (int i=1; i<2*norbs; i++)
    Jred(i + 2*norbs -1) = JA(index(i,0));
  Jred(4*norbs -1) = JA(index(2,1));
  //Jred(4*norbs) = JA(index(3,1));
           
  for (int ind = 0; ind<NonRedundantMap.size(); ind++) {
    int i = NonRedundantMap[ind].first,
        j = NonRedundantMap[ind].second;
    JnonRed(ind) = JA(index(i,j));
  }
};

template<typename T>
void JastrowFromRedundantAndNonRedundant(Matrix<T, Dynamic, 1>& JA, Matrix<T, Dynamic, 1>& Jred,
                                         Matrix<T, Dynamic, 1>& JnonRed,
                                         vector<pair<int, int>>& NonRedundantMap) {
  int norbs = Determinant::norbs;

  //diagonal jastrow are redundant
  for (int i=0; i<2*norbs; i++)
    JA(index(i,i)) = Jred(i);
  for (int i=1; i<2*norbs; i++)
    JA(index(i,0)) = Jred(i + 2*norbs -1);
  JA(index(2,1)) = Jred(4*norbs -1);
  //JA(index(3,1)) = Jred(4*norbs);

  for (int ind = 0; ind<NonRedundantMap.size(); ind++) {
    int i = NonRedundantMap[ind].first,
        j = NonRedundantMap[ind].second;
    JA(index(i,j)) = JnonRed(ind);
  }

};



//term is orb1^dag orb2
template<typename T>
T getCreDesDiagMatrix(DiagonalMatrix<T, Dynamic>& diagcre,
                      DiagonalMatrix<T, Dynamic>& diagdes,
                      int orb1,
                      int orb2,
                      int norbs,
                      const Matrix<T, Dynamic, 1>&JA)  {
  
  for (int j=0; j<2*norbs; j++) {
    diagcre.diagonal()[j] = exp(-2.*JA(index(orb1, j)));
    diagdes.diagonal()[j] = exp( 2.*JA(index(orb2, j)));
  }

  T factor = exp( JA(index(orb1, orb1)) - JA(index(orb2, orb2)));
  return factor;
};

template<typename T>
T getCreDesDiagMatrix(DiagonalMatrix<T, Dynamic>& diagcre,
                      DiagonalMatrix<T, Dynamic>& diagdes,
                      int orb1,
                      int orb2,
                      int orb3,
                      int orb4,
                      int norbs,
                      const Matrix<T, Dynamic, 1>&JA)  {

  if (orb3 == -1 && orb4 == -1)
    return getCreDesDiagMatrix(diagcre, diagdes, orb1, orb2, norbs, JA);
  
  for (int j=0; j<2*norbs; j++) {
    diagcre.diagonal()[j] = exp(-2.*JA(index(orb1, j)) - 2.*JA(index(orb2, j)));
    diagdes.diagonal()[j] = exp(+2.*JA(index(orb3, j)) + 2.*JA(index(orb4, j)));
  }

  T factor = exp( 2*JA(index(orb1, orb2)) + JA(index(orb1, orb1)) + JA(index(orb2, orb2))
                  -2*JA(index(orb3, orb4)) - JA(index(orb3, orb3)) - JA(index(orb4, orb4))) ;
  return factor;
};

template<typename T, typename complexT>
void applyProjector(
    Matrix<complexT, Dynamic, Dynamic>& bra,
    vector<Matrix<complexT, Dynamic, Dynamic>>& ketvec,
    vector<complexT>& coeffs,
    T Sz,
    int ngrid) {
  
  int norbs = Determinant::norbs;
  T m = 1.*Sz/2.;
  int count = 0;
  complexT iImag(0, 1.0);
  
  auto CoeffF =  [&, count](complexT&) mutable -> complexT
      {
        complexT v = -1.* iImag * (1. *(((++count) - 1)/2) * 2 * M_PI * m/ngrid);
        //complexT val = exp(iImag);// * (1. *(((++count) - 1)/2) * 2 * M_PI * m/ngrid))/(2.*ngrid);
        complexT val(exp(v.real())*cos(v.imag()), exp(v.real())*sin(v.imag()));
        return val/(2.*ngrid);
      };
  
  auto ketF = [&, count](Matrix<complexT, Dynamic, Dynamic>&)
      mutable -> Matrix<complexT, Dynamic, Dynamic>
      {
        Matrix<complexT, Dynamic, 1> phi = Matrix<complexT, Dynamic, 1>::Ones(2*norbs);
        complexT vp = iImag * (1.*count * 2 * M_PI / ngrid /2.);
        complexT vm = -iImag * (1.*count * 2 * M_PI / ngrid /2.);
        phi.segment(0,norbs)     *= complexT(exp(vp.real())*cos(vp.imag()),
                                             exp(vp.real())*sin(vp.imag()));
        phi.segment(norbs,norbs) *= complexT(exp(vm.real())*cos(vm.imag()),
                                             exp(vm.real())*sin(vm.imag()));
        count ++;
        Matrix<complexT, Dynamic, Dynamic> mat = phi.asDiagonal()*bra;
        if (count%2 == 0) return mat;
        else return mat.conjugate();
        //return phi.asDiagonal()*bra;
      };

  coeffs.resize(2*ngrid);
  std::transform(coeffs.begin(), coeffs.end(), coeffs.begin(),
                 CoeffF);


  ketvec.resize(2*ngrid);
  std::transform(ketvec.begin(), ketvec.end(), ketvec.begin(),
                 ketF);


}

template<typename T>
void fillJastrowfromWfn(Matrix<double, Dynamic, Dynamic>& Jtmp, Matrix<T, Dynamic, 1>& JA) {
  
  int norbs = Determinant::norbs;

  auto f = [&] (int i, int j) {
    int I = ABAB(i), J = ABAB(j);
    JA(index(i,j)) =  log(Jtmp(max(I,J), min(I,J)));
    JA(index(i,j)) /=  i==j ? 1.: 2.;
  };
  
  loopOverLowerTriangle(f);
  
};


template<typename T>
void fillWfnfromJastrow(Matrix<T, Dynamic, 1>& JA,
                        Matrix<T, Dynamic, Dynamic>& Jtmp) {
  int norbs = Determinant::norbs;
  
  auto f = [&] (int i, int j) {
    int I = ABAB(i), J = ABAB(j);
    Jtmp(max(I,J), min(I,J)) = i == j ? exp(JA(index(i,j))) : exp(2.*JA(index(i,j)));
  };
  
  loopOverLowerTriangle(f);
  
};

template<typename T, typename complexT>
Matrix<complexT, Dynamic, Dynamic> fillWfnOrbs(Matrix<complexT, Dynamic, Dynamic>& orbitals,
                                               Matrix<T, Dynamic, 1>& variables)  {
  int norbs  = Determinant::norbs;
  int nalpha = Determinant::nalpha;
  int nbeta  = Determinant::nbeta;
  int nelec =  nalpha + nbeta;

  //Use the rotation matrix variables and update the bra
  Matrix<complexT, Dynamic, Dynamic> U(2*norbs-nelec, nelec);
  for (int a=0; a<2*norbs-nelec; a++) 
    for (int i=0; i<nelec; i++) {
      U(a,i).real(variables( 2* (a*nelec+i)  ));
      U(a,i).imag(variables( 2* (a*nelec+i)+1));
      
      //U(a, i) = complexT( variables( 2* (a*nelec+i)  ),
      //variables( 2* (a*nelec+i)+1)
      //);
    }
  
  Matrix<complexT, Dynamic, Dynamic> bra = orbitals.block(0, 0, 2*norbs, nelec)
      + orbitals.block(0, nelec, 2*norbs, 2*norbs-nelec) * U;
  return bra;
};


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
    vector<MatrixXcT> ket; vector<complexT> coeffs;
    T Sz = 1.*(nalpha-nbeta);
    applyProjector(bra, ket, coeffs, Sz, ngrid); 


    MatrixXcT S;
    T detovlp = 0.;

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
    calcRDM<complexT> calcrdm;   
    MatrixXcT LambdaD, LambdaC, S;
    DiagonalXT diagcre(2*norbs),
        diagdes(2*norbs);  
    MatrixXcT rdm, Sinv, JJphi, JJphiSinv;
    complexT Energy, Sdet, factor;

    //calcualte the residual for gradient
    auto orbGrad = [&] (const int& orb1, const int& orb2, const complexT& f) {
      orbitalResidueMat.row(orb1) += f * (diagcre.diagonal()[orb1] * (LambdaD.row(orb2) * Sinv));
      orbitalResidueMat += f * (-((JJphiSinv*LambdaC.adjoint().col(orb1)) * (LambdaD.row(orb2) * Sinv)));
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
      loopOverLowerTriangle(Jres);
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
    
      size_t osize = 2*orbitalResidueMat.rows() * orbitalResidueMat.cols();
      MPI_Allreduce(MPI_IN_PLACE, &orbitalResidueMat(0,0), osize, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
      size_t two = 2;
      MPI_Allreduce(MPI_IN_PLACE, &Energy, two, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }
#endif

    return Energy.real();
  };

  T GetResidual::getJastrowResidue(const VectorXT& jastrowvars,
                                   const VectorXT& braVars,
                                   VectorXT& jastrowResidue) {
    int nJastrowVars = jastrowvars.size(),
        nBraVars = braVars.size();
    VectorXT variables(nJastrowVars+nBraVars);

    variables.block(0, 0, nJastrowVars, 1) = jastrowvars;
    variables.block(nJastrowVars, 0, nBraVars, 1) = braVars;

    VectorXT residue = variables;
    MatrixXT JastrowHessian;
    T Energy = getResidue(variables, residue, JastrowHessian, true, false, false);

    jastrowResidue = residue.block(0, 0, nJastrowVars, 1);
    return Energy;
  }

  T GetResidual::getOrbitalResidue(const VectorXT& jastrowvars,
                                   const VectorXT& braVars,
                                   VectorXT& orbitalResidue) {

    int nJastrowVars = jastrowvars.size(),
        nBraVars = braVars.size();
    VectorXT variables(nJastrowVars+nBraVars);

    variables.block(0, 0, nJastrowVars, 1) = jastrowvars;
    variables.block(nJastrowVars, 0, nBraVars, 1) = braVars;

    VectorXT residue = variables;
    MatrixXT JastrowHessian;
    T Energy = getResidue(variables, residue, JastrowHessian, false, true, false);

    orbitalResidue = residue.block(nJastrowVars, 0, nBraVars, 1);
    return Energy;
  }
  
};





