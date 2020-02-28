#pragma once

#include "ResidualsUtils.h"
#include "calcRDM.h"
#include <taco.h>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace std;
using namespace Eigen;

template<typename T, typename complexT>
struct NociResidual {
  using MatrixXcT = Matrix<complexT, Dynamic, Dynamic>;
  using VectorXcT = Matrix<complexT, Dynamic, 1>;
  using MatrixXT = Matrix<T, Dynamic, Dynamic>;
  using VectorXT = Matrix<T, Dynamic, 1>;
  using DiagonalXT = Eigen::DiagonalMatrix<T, Eigen::Dynamic>;

  using Tensor4d = taco::Tensor<complexT>;

  int ngrid;
  int norbs;
  taco::Tensor<complexT> temp, AARDM, AAGrad, ABRDM, ABGrad, BBRDM, BBGrad, BARDM, BAGrad, AABBRDM;
  taco::Tensor<complexT> out, out1, out2, out3, out4;
  taco::Tensor<double>& Int2e;
  NociResidual(int pnorbs, taco::Tensor<double>& ptwoInt, int pngrid=4)
      : Int2e(ptwoInt), norbs(pnorbs), ngrid(pngrid)
  {
    taco::IndexVar I, J, K, L;
    taco::Format D4d({taco::Dense, taco::Dense, taco::Dense, taco::Dense});
    taco::Format D2d({taco::Dense, taco::Dense});
  
    out =taco::Tensor<complexT> ("o"); 
    out1=taco::Tensor<complexT> ("o1"); 
    out2=taco::Tensor<complexT> ("o2"); 
    out3=taco::Tensor<complexT> ("o3"); 
    out4=taco::Tensor<complexT> ("o4"); 

    temp=taco::Tensor<complexT> ({norbs, norbs}, D2d);
    AARDM=taco::Tensor<complexT>({norbs, norbs}, D2d);
    AAGrad=taco::Tensor<complexT> ({norbs, norbs}, D2d);
    ABRDM=taco::Tensor<complexT> ({norbs, norbs}, D2d);
    ABGrad=taco::Tensor<complexT>  ({norbs, norbs}, D2d);
    BBRDM=taco::Tensor<complexT> ({norbs, norbs}, D2d);
    BBGrad=taco::Tensor<complexT> ({norbs, norbs}, D2d);
    BARDM=taco::Tensor<complexT> ({norbs, norbs}, D2d);
    BAGrad=taco::Tensor<complexT> ({norbs, norbs}, D2d);
    AABBRDM=taco::Tensor<complexT> ({norbs, norbs}, D2d);

    //TACO expression precompiled
    AABBRDM(I,J) = AARDM(I,J) + BBRDM(I, J); AABBRDM.compile(); AABBRDM.assemble();

    AAGrad(J, I) = (Int2e(I, J, K, L) * AABBRDM(L, K)) - (Int2e(I, L, K, J) * AARDM(L, K));
    AAGrad.compile(); AAGrad.assemble(); 

    BBGrad(J, I) = (Int2e(I, J, K, L) * AABBRDM(L, K)) - (Int2e(I, L, K, J) * BBRDM(L, K));
    BBGrad.compile(); BBGrad.assemble(); 

    ABGrad(L, I) =    - (Int2e(I, J, K, L) * BARDM(J, K)) - (Int2e(K, L, I, J) * BARDM(J, K));
    ABGrad.compile(); ABGrad.assemble(); 

    BAGrad(L, I) =    - (Int2e(I, J, K, L) * ABRDM(J, K)) - (Int2e(K, L, I, J) * ABRDM(J, K));
    BAGrad.compile(); BAGrad.assemble();


    out1 = AARDM(J, I) * AAGrad(J, I); out1.compile(); out1.assemble();
    out2 = ABRDM(J, I) * ABGrad(J, I); out2.compile(); out2.assemble();
    out3 = BARDM(J, I) * BAGrad(J, I); out3.compile(); out3.assemble();
    out4 = BBRDM(J, I) * BBGrad(J, I); out4.compile(); out4.assemble();

    //
  };

  T getLagrangianNoJastrow(const MatrixXcT& bra,
                           const vector<MatrixXcT>& ketvec,
                           MatrixXcT& NumGrad,
                           MatrixXcT& DenGrad,
                           VectorXd& NumVec,
                           VectorXd& DenVec,
                           MatrixXd& oneInt) {

    
    int norbs  = Determinant::norbs;
    int nalpha = Determinant::nalpha;
    int nbeta  = Determinant::nbeta;
    int nelec =  nalpha + nbeta;

    NumVec.setZero(); DenVec.setZero();
    
    //apply the Sz projector and generate a linear combination of kets
    vector<MatrixXcT> ket; vector<complexT> coeffs;
    vector<VectorXcT> Projector;
    T Sz = 1.*(nalpha-nbeta);

    //add projected bra to the ket
    for (int i=0; i<ketvec.size(); i++) {
      applyProjector(ketvec[i], ket, coeffs, Projector, Sz, ngrid);
    }

    MatrixXcT S;
    MatrixXcT mfRDM(2*norbs, 2*norbs);


    MatrixXcT RDMGradE(2*norbs, 2*norbs);
    RDMGradE.setZero();

    
    //oneRDM.resize(2*norbs, 2*norbs);
    for (int g = commrank; g<ket.size(); g+=commsize) {
      S = bra.adjoint()*ket[g] ;
      MatrixXcT Sinv = S.inverse();
      complexT Sdet = S.determinant();
      complexT factor = Sdet *coeffs[g];
      complexT Eiter = 0.0;
      
      mfRDM = (((ket[g] * Sinv)*bra.adjoint()));

      for (int i=0; i<norbs; i++)
        for (int j=0; j<norbs; j++) {
          AARDM.insert({i,j}, mfRDM(j, i));
          ABRDM.insert({i,j}, mfRDM(j, i+norbs));
          BARDM.insert({i,j}, mfRDM(j+norbs, i));
          BBRDM.insert({i,j}, mfRDM(j+norbs, i+norbs));
        }
      AARDM.pack(), ABRDM.pack(), BARDM.pack(), BBRDM.pack();
      
      Eiter += factor * (mfRDM.block(0,0,norbs, norbs).cwiseProduct(oneInt)).sum();
      Eiter += factor * (mfRDM.block(norbs,norbs,norbs, norbs).cwiseProduct(oneInt)).sum();
      RDMGradE.block(0,0,norbs, norbs) += oneInt.transpose() * factor;
      RDMGradE.block(norbs,norbs,norbs, norbs) += oneInt.transpose() * factor;

      

      AABBRDM.compute();
      AAGrad.compute(); BAGrad.compute(); ABGrad.compute(); BBGrad.compute();
      out1.compute(); out2.compute(); out3.compute(); out4.compute();
      //out.compute();

      Eiter += 0.5*factor*( static_cast<const complexT*>(out1.getStorage().getValues().getData())[0]
                          +0.5*static_cast<const complexT*>(out2.getStorage().getValues().getData())[0]
                          +0.5*static_cast<const complexT*>(out3.getStorage().getValues().getData())[0]
                          +static_cast<const complexT*>(out4.getStorage().getValues().getData())[0]);

      for (int i=0; i<norbs; i++)
        for (int j=0; j<norbs; j++) {
          RDMGradE(j      , i      ) += 1.00 * factor * static_cast<const complexT*>(AAGrad.getStorage().getValues().getData())[i*norbs+j];
          RDMGradE(j      , i+norbs) += 0.50 * factor * static_cast<const complexT*>(ABGrad.getStorage().getValues().getData())[i*norbs+j];
          RDMGradE(j+norbs, i      ) += 0.50 * factor * static_cast<const complexT*>(BAGrad.getStorage().getValues().getData())[i*norbs+j];
          RDMGradE(j+norbs, i+norbs) += 1.00 * factor * static_cast<const complexT*>(BBGrad.getStorage().getValues().getData())[i*norbs+j];
        }

      
      /*
      auto twoRDM = [&](const int& i, const int& j, const int& k,
                        const int& l, const double& i2) {
        auto val = i2 * factor * (mfRDM(l, i)*mfRDM(k, j) - mfRDM(k, i)*mfRDM(l, j));

        RDMGradE(l, i) += i2 * factor * mfRDM(k, j);
        RDMGradE(k, j) += i2 * factor * mfRDM(l, i);
        RDMGradE(k, i) -= i2 * factor * mfRDM(l, j);
        RDMGradE(l, j) -= i2 * factor * mfRDM(k, i);
        Eiter += val;        
      };
      //loopOver2epar(twoRDM);
      loopOver2e(twoRDM);
      //cout << Eiter<<endl;exit(0);
      */
      //exit(0);
      
      //Derivative w.r.t. BRA
      {      
        NumGrad +=  RDMGradE.transpose() * ket[g] * Sinv;
        NumGrad -=  mfRDM * RDMGradE.transpose() * ket[g] * Sinv;
        NumGrad +=  (ket[g]*Sinv) * (Eiter);
        
        DenGrad +=  (ket[g]*Sinv) * Sdet * coeffs[g];
      }

      
      
      RDMGradE.setZero();

      NumVec[g / (2*ngrid)] += Eiter.real();
      DenVec[g / (2*ngrid)] += (Sdet*coeffs[g]).real();
    }
    
#ifndef SERIAL
    size_t size = NumVec.size();
    MPI_Allreduce(MPI_IN_PLACE, &NumVec[0], size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &DenVec[0], size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    size_t jsize = 2*NumGrad.size();
    MPI_Allreduce(MPI_IN_PLACE, &NumGrad(0,0), jsize, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &DenGrad(0,0), jsize, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);      
#endif
    
    T Lagrangian = NumVec.sum();
    
    return (Lagrangian/DenVec.sum() + coreE);
    
  };

};
