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
void loopOverAlphaOrbs(F& fun){

  int norbs = Determinant::norbs;
  for (int orb1 = 0; orb1 <  norbs ; orb1++) {
    fun(orb1);
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

template<typename F>
void loopOverGutzwillerJastrow(F& fun) {
  
  int norbs = Determinant::norbs;
  for (int orbn = 0; orbn < norbs; orbn++) 
    fun(orbn+norbs, orbn);
};


void ConstructRedundantJastrowMap(vector<pair<int,int>>& NonRedundantMap) ;

template<typename T>
void RedundantAndNonRedundantJastrow(Matrix<T, Dynamic, 1>& JA, Matrix<T, Dynamic, 1>& Jred,
                                     Matrix<T, Dynamic, 1>& JnonRed,
                                     vector<pair<int, int>>& NonRedundantMap) {
  
  int norbs = Determinant::norbs;
  
  int ind1 = 0, ind2 = 0;
  for (int i=0; i<2*norbs; i++)
    for (int j=0; j<=i; j++) {
      if (NonRedundantMap[ind1].first == i &&
          NonRedundantMap[ind1].second == j) {
        ind1++;
      }
      else {
        Jred(ind2) = JA(index(i,j)); 
        ind2 ++;
      }
    }

           
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

  int ind1 = 0, ind2 = 0;
  for (int i=0; i<2*norbs; i++)
    for (int j=0; j<=i; j++) {
      if (NonRedundantMap[ind1].first == i &&
          NonRedundantMap[ind1].second == j)
        ind1++;
      else {
        JA(index(i,j)) = Jred(ind2) ; 
        ind2 ++;
      }
    }
  
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
    const Matrix<complexT, Dynamic, Dynamic>& bra,
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
        complexT v = -(1.* iImag * 2. * M_PI * m * (count/2))/ngrid;
        //complexT v = -(1.* iImag * 2. * M_PI * m * (count))/ngrid;
        complexT val(exp(v.real())*cos(v.imag()), exp(v.real())*sin(v.imag()));

        count++;

        
        if ((count-1)%2 == 0) return val/(sqrt(2.)*ngrid);
        else return conj(val)/(sqrt(2.)*ngrid);
      };
  
  auto ketF = [&, count](Matrix<complexT, Dynamic, Dynamic>&)
      mutable -> Matrix<complexT, Dynamic, Dynamic>
      {
        Matrix<complexT, Dynamic, 1> phi = Matrix<complexT, Dynamic, 1>::Ones(2*norbs);
        complexT vp = iImag * (1.* (count/2) * 2 * M_PI) / ngrid ;
        complexT vm = -iImag * (1.* (count/2) * 2 * M_PI) / ngrid ;
        phi.segment(0,norbs)     *= complexT(exp(vp.real())*cos(vp.imag()),
                                             exp(vp.real())*sin(vp.imag()));
        phi.segment(norbs,norbs) *= complexT(exp(vm.real())*cos(vm.imag()),
                                             exp(vm.real())*sin(vm.imag()));
        Matrix<complexT, Dynamic, Dynamic> mat = phi.asDiagonal()*bra;
        count ++;

        if ((count-1)%2 == 0) return mat;
        else  return mat.conjugate();
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

