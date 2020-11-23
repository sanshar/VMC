/*
  Developed by Sandeep Sharma
  Copyright (c) 2017, Sandeep Sharma
  
  This file is part of DICE.
  
  This program is free software: you can redistribute it and/or modify it under the terms
  of the GNU General Public License as published by the Free Software Foundation, 
  either version 3 of the License, or (at your option) any later version.
  
  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
  
  See the GNU General Public License for more details.
  
  You should have received a copy of the GNU General Public License along with this program. 
  If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef EvalE_HEADER_H
#define EvalE_HEADER_H
#include <Eigen/Dense>
#include <vector>
#include "Determinants.h"
#include "rDeterminants.h"
#include "workingArray.h"
#include "statistics.h"
#include "sr.h"
#include "linearMethod.h"
#include "global.h"
#include "Deterministic.h"
#include "ContinuousTime.h"
#include "Metropolis.h"
#include <iostream>
#include <fstream>
#include <boost/serialization/serialization.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <algorithm>
#include "boost/format.hpp"

#ifndef SERIAL
#include "mpi.h"
#endif

using namespace Eigen;
using namespace std;
using namespace boost;

class oneInt;
class twoInt;
class twoIntHeatBathSHM;

//############################################################Deterministic Evaluation############################################################################

void updateTwoRdm(int i, int j, int a, int b, int norbs, double contribution,
                  MatrixXd& twoRdm) {
  
  //spin to spatial
  //int I = i/2, J = j/2, B = b/2, A = a/2;
  int I = i/2, J = j/2, A = a/2, B = b/2;

  twoRdm ( I*norbs+J, A*norbs+B) +=  contribution;
  twoRdm ( J*norbs+I, B*norbs+A) +=  contribution;

  if (i%2 == j%2) {
    twoRdm ( J*norbs+I, A*norbs+B) += -contribution;
    twoRdm ( I*norbs+J, B*norbs+A) += -contribution;
  }
}

template<typename Wfn, typename Walker>
void getEnergyDeterministic(Wfn &w, Walker& walk, double &Energy)
{
  Deterministic<Wfn, Walker> D(w, walk);
  Energy = 0.0;
  for (int i = commrank; i < D.allDets.size(); i += commsize)
  {
    D.LocalEnergy(D.allDets[i]);
    D.UpdateEnergy(Energy);
  }
  D.FinishEnergy(Energy);
}

template<typename Wfn, typename Walker>
void getTransEnergyDeterministic(Wfn &w, Walker& walk, double &Energy)
{
  Deterministic<Wfn, Walker> D(w, walk);
  double Numerator =0., Denominator = 0.;
  double rdm = 0.;
  int orb1, orb2;
  int norbs = Determinant::norbs;
  VectorXd NiNjRDMH(2*norbs*(2*norbs+1)/2); NiNjRDMH.setZero();
  VectorXd NiNjRDM(2*norbs*(2*norbs+1)/2); NiNjRDM.setZero();

  for (int i = commrank; i < D.allDets.size(); i += commsize)
  {
    double ovlp = 0.0, Eloc = 0.0;
    w.initWalker(walk, D.allDets[i]);
    w.HamAndOvlp(walk, ovlp, Eloc, D.work, false);  
    double CorrOvlp = w.corr.Overlap(D.allDets[i]);
    double SlaterOvlp = walk.getDetOverlap(w.getRef());

    for (int orb1 = 0; orb1<2*norbs; orb1++)
      for (int orb2 = 0; orb2<=orb1; orb2++)
        if (D.allDets[i].getocc(orb1%norbs, orb1/norbs) &&
            D.allDets[i].getocc(orb2%norbs, orb2/norbs)) {
          NiNjRDMH(orb1*(orb1+1)/2 + orb2) += (Eloc * ovlp) * SlaterOvlp/CorrOvlp;
          NiNjRDM(orb1*(orb1+1)/2 + orb2) += SlaterOvlp * SlaterOvlp;
        }
          /*
    if (D.allDets[i].getoccA(orb1) && !D.allDets[i].getoccA(orb2)) {
      Determinant dtemp = D.allDets[i];
      dtemp.setoccA(orb1, false); dtemp.setoccA(orb2, true);
      w.initWalker(walk, dtemp);

      double CorrOvlptmp = w.corr.Overlap(dtemp);
      double SlaterOvlptmp = walk.getDetOverlap(w.getRef());
      
      rdm += ( CorrOvlptmp*SlaterOvlptmp) * SlaterOvlp/CorrOvlp;
    }
    */
    Numerator += (Eloc * ovlp) * SlaterOvlp/CorrOvlp;
    Denominator += SlaterOvlp * SlaterOvlp;
  }

#ifndef SERIAL
  int sizeninj = 2*norbs*(2*norbs+1)/2;
  MPI_Allreduce(MPI_IN_PLACE, &(NiNjRDM(0)), sizeninj, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(NiNjRDMH(0)), sizeninj, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(rdm), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(Numerator), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(Denominator), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
  if (commrank == 0) cout << Denominator<<"  "<<Numerator<<endl;
  if (commrank == 0) cout << Denominator<<"  "<<rdm<<"  "<<rdm/Denominator<<endl;
  Energy = Numerator/Denominator;
  NiNjRDM *= Energy/Denominator;
  NiNjRDMH /= Denominator;
  //if (commrank == 0) cout << "ninj"<<endl<<NiNjRDM<<endl;
  //if (commrank == 0) cout << "ninjh"<<endl<<NiNjRDMH<<endl;

}


template<typename Wfn, typename Walker> void getOneRdmDeterministic(Wfn &w, Walker& walk, MatrixXd &oneRdm, bool sz)
{
  int norbs = Determinant::norbs;
  int nalpha = Determinant::nalpha;
  int nbeta = Determinant::nbeta;

  vector<Determinant> allDets;
  generateAllDeterminants(allDets, norbs, nalpha, nbeta);

  double Overlap = 0;
  oneRdm = MatrixXd::Constant(norbs, norbs, 0.); 
  MatrixXd localOneRdm = MatrixXd::Constant(norbs, norbs, 0.); 

  for (int i = commrank; i < allDets.size(); i += commsize) {
    w.initWalker(walk, allDets[i]);
    localOneRdm.setZero(norbs, norbs);
    vector<int> open;
    vector<int> closed;
    allDets[i].getOpenClosed(sz, open, closed);
    for (int p = 0; p < closed.size(); p++) {
      localOneRdm(closed[p], closed[p]) = 1.;
      for (int q = 0; q < open.size() && open[q] < closed[p]; q++) {
        localOneRdm(closed[p], open[q]) = w.getOverlapFactor(2*closed[p] + sz, 2*open[q] + sz, walk, 0);
      }
    }
    double ovlp = w.Overlap(walk);
    Overlap += ovlp * ovlp;
    oneRdm += ovlp * ovlp * localOneRdm;
  }
#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &(Overlap), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, oneRdm.data(), norbs*norbs, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  oneRdm = oneRdm / Overlap;
};

template<typename Wfn, typename Walker>
void getTwoRdmDeterministic(Wfn &w, Walker& walk, MatrixXd &twoRdm)
{
  int norbs = Determinant::norbs;
  int nalpha = Determinant::nalpha;
  int nbeta = Determinant::nbeta;
  int nelec = nalpha + nbeta;
  
  vector<Determinant> allDets;
  generateAllDeterminants(allDets, norbs, nalpha, nbeta);

  double Overlap = 0;
  twoRdm = MatrixXd::Constant(norbs*norbs, norbs*norbs, 0.); 

  double a1=0, a2=0;
  for (int i = commrank; i < allDets.size(); i += commsize) {
    w.initWalker(walk, allDets[i]);
    double ovlp = w.Overlap(walk);
    Overlap += ovlp * ovlp;
    
    vector<int> open;
    vector<int> closed;
    allDets[i].getOpenClosed(open, closed);

    double contribution = ovlp*ovlp;
    
    //diagonal elements
    for (int a = 0; a < closed.size(); a++) 
      for (int b = 0; b < closed.size(); b++)  {
        if (a == b) continue;
        int A = closed[a]/2, B = closed[b]/2;
        twoRdm(A*norbs + B, B*norbs +A) += contribution * 0.5;
        twoRdm(B*norbs + A, A*norbs +B) += contribution * 0.5;
        if (closed[a]%2 == closed[b]%2) {
          twoRdm(A*norbs + B, A*norbs +B) += -contribution * 0.5;
          twoRdm(B*norbs + A, B*norbs +A) += -contribution * 0.5;
        }
      }


    //singleExcitations contribution
    for (int a = 0; a < closed.size(); a++) 
      for (int i = 0; i < open.size(); i++) {
        if (closed[a]%2 != open[i]%2) continue; 
        double ovlpratio = w.getOverlapFactor(closed[a], open[i], walk, 0);
        for (int b = 0; b < closed.size(); b++) {          
          if (a == b) continue;          
          updateTwoRdm(closed[a], closed[b], closed[b], open[i], norbs, contribution*ovlpratio, twoRdm);          
        }
    }

    
    //twordm contribution
    for (int a = 0; a < closed.size(); a++) 
      for (int b = a+1; b < closed.size() ; b++)  {
        for (int i = 0; i < open.size(); i++) 
          for (int j = i+1; j < open.size() ; j++) {
            if (closed[a]%2+closed[b]%2 - open[i]%2-open[j]%2 == 0) {
              int A = closed[a], B = closed[b], I = open[i], J = open[j];
              if (A%2 != I%2) {J = open[i]; I = open[j];}
              
              double ovlpratio = w.getOverlapFactor(A, B, I, J, walk, 0);
              updateTwoRdm(A, B, J, I, norbs, contribution*ovlpratio, twoRdm);
            }
          }
      }

  }
#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &(Overlap), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, twoRdm.data(), norbs*norbs, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
  twoRdm = twoRdm / Overlap;

  if (commrank == 0) {
    char file[5000];
    int root = 0;
    sprintf(file, "spatialRDM.%d.%d.txt", root,
            root);
    std::ofstream ofs(file, std::ios::out);
    ofs << norbs << endl;
    double Etwo = 0.0, Eone = 0.0;
    for (int i=0; i<norbs; i++)
      for (int j=0; j<norbs; j++)
        for (int a=0; a<norbs; a++)
          for (int b=0; b<norbs; b++) {
            ofs << str(boost::format("%3d   %3d   %3d   %3d   %10.8g\n") %
                       i % j % b % a %
                       (twoRdm(i * norbs + j, a * norbs + b)));
            //double int2 = I2 (2*i, 2*b, 2*j, 2*a);
            //Etwo += 0.5 * twoRdm(i*norbs + j , a*norbs + b) * int2;
          }
    ofs.close();
    
  }
  MPI_Barrier(MPI_COMM_WORLD);
  /*
  double Etwo = 0.0, Eone = 0.0;
  for (int i=0; i<norbs; i++)
    for (int j=0; j<norbs; j++)
      for (int a=0; a<norbs; a++)
        for (int b=0; b<norbs; b++) {
          //cout << i<<"  "<<j<<"  "<<a<<"  "<<b<<"  "<<twoRdm(i*norbs+j, a*norbs+b)/2.<<endl;
          double int2 = I2 (2*i, 2*b, 2*j, 2*a);
          Etwo += 0.5 * twoRdm(i*norbs + j , a*norbs + b) * int2;
        }

  MatrixXd oneRDM = MatrixXd::Constant(norbs, norbs, 0.0);
  for (int i=0; i<norbs; i++)
    for (int j=0; j<norbs; j++) {
      for (int k=0; k<norbs; k++) 
        oneRDM(i, j) += twoRdm(i*norbs+k, k*norbs+j)/(nelec-1);
      double int1 = I1(2*i, 2*j);
      Eone += oneRDM(i, j) * int1;
    }
  cout << oneRDM<<endl;
  cout << "Energy "<< Etwo<<"  "<<Eone<<endl;
  cout << Etwo + Eone + coreE<<endl;
  exit(0);
  */
}

template<typename Wfn, typename Walker> void getDensityCorrelationsDeterministic(Wfn &w, Walker& walk, MatrixXd &corr)
{
  int norbs = Determinant::norbs;
  int nalpha = Determinant::nalpha;
  int nbeta = Determinant::nbeta;

  vector<Determinant> allDets;
  generateAllDeterminants(allDets, norbs, nalpha, nbeta);

  double Overlap = 0;
  corr = MatrixXd::Constant(2*norbs, 2*norbs, 0.);
  MatrixXd localCorr = MatrixXd::Constant(2*norbs, 2*norbs, 0.); 

  for (int i = commrank; i < allDets.size(); i += commsize) {
    w.initWalker(walk, allDets[i]);
    localCorr.setZero(2*norbs, 2*norbs);
    vector<int> open;
    vector<int> closed;
    allDets[i].getOpenClosed(open, closed);
    for (int p = 0; p < closed.size(); p++) {
      localCorr(closed[p], closed[p]) = 1.;
      for (int q = 0; q < p; q++) {
        int P = max(closed[p], closed[q]), Q = min(closed[p], closed[q]);
        localCorr(P, Q) = 1.;
      }
    }
    double ovlp = w.Overlap(walk);
    Overlap += ovlp * ovlp;
    corr += ovlp * ovlp * localCorr;
  }
#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &(Overlap), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, corr.data(), 4*norbs*norbs, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  corr = corr / Overlap;
}
  
template<typename Wfn, typename Walker>
void getGradientDeterministic(Wfn &w, Walker &walk, double &Energy, VectorXd &grad)
{
  Deterministic<Wfn, Walker> D(w, walk);
  Energy = 0.0;
  grad.setZero();
  VectorXd grad_ratio_bar = VectorXd::Zero(grad.rows());
  for (int i = commrank; i < D.allDets.size(); i += commsize)
  {
    //if (schd.debug) cout << "det   " << D.allDets[i] << endl;
    D.LocalEnergy(D.allDets[i]);
    D.LocalGradient();
    D.UpdateEnergy(Energy);
    D.UpdateGradient(grad, grad_ratio_bar);
  }
  D.FinishEnergy(Energy); 
  D.FinishGradient(grad, grad_ratio_bar, Energy);
}

template<typename Wfn, typename Walker>
void getGradientMetricDeterministic(Wfn &w, Walker &walk, double &Energy, VectorXd &grad, VectorXd &H, DirectMetric &S)
{
  Deterministic<Wfn, Walker> D(w, walk);
  Energy = 0.0;
  grad.setZero();
  VectorXd grad_ratio_bar = VectorXd::Zero(grad.rows());
  for (int i = commrank; i < D.allDets.size(); i += commsize) 
  {
    D.LocalEnergy(D.allDets[i]);
    D.LocalGradient();
    D.UpdateEnergy(Energy);
    D.UpdateGradient(grad, grad_ratio_bar);
    D.UpdateSR(S);
  }
  D.FinishEnergy(Energy);
  D.FinishGradient(grad, grad_ratio_bar, Energy);
  D.FinishSR(grad, grad_ratio_bar, H, S);
}

template<typename Wfn, typename Walker>
void getGradientHessianDeterministic(Wfn &w, Walker& walk, double &Energy, VectorXd &grad, MatrixXd& Hmatrix, MatrixXd &Smatrix)
{
  Deterministic <Wfn, Walker> D(w, walk);
  int numVars = grad.rows();
  Energy = 0.0;
  grad.setZero();
  Eigen::VectorXd grad_ratio_bar = VectorXd::Zero(numVars);
  Hmatrix = MatrixXd::Zero(numVars + 1, numVars+1);
  Smatrix = MatrixXd::Zero(numVars + 1, numVars+1);
  for (int i = commrank; i < D.allDets.size(); i += commsize)
  {
    D.LocalEnergy(D.allDets[i]);
    D.LocalGradient();
    D.LocalEnergyGradient();
    D.UpdateEnergy(Energy);
    D.UpdateGradient(grad, grad_ratio_bar);
    D.UpdateLM(Hmatrix, Smatrix);
  }
  D.FinishEnergy(Energy);
  D.FinishGradient(grad, grad_ratio_bar, Energy);
  D.FinishLM(Hmatrix, Smatrix);
}

template<typename Wfn, typename Walker>
void getGradientHessianDirectDeterministic(Wfn &w, Walker& walk, double &Energy, VectorXd &grad, DirectLM &h)
{
  Deterministic<Wfn, Walker> D(w, walk);
  int numVars = grad.rows();
  Energy = 0.0;
  grad.setZero();
  Eigen::VectorXd grad_ratio_bar = VectorXd::Zero(numVars);
  h.T.clear();
  h.G.clear();
  h.H.clear();
  for (int i = commrank; i < D.allDets.size(); i += commsize)
  {
    D.LocalEnergy(D.allDets[i]);
    D.LocalGradient();
    D.LocalEnergyGradient();
    D.UpdateEnergy(Energy);
    D.UpdateGradient(grad, grad_ratio_bar);
    D.UpdateLM(h);
  }
  D.FinishEnergy(Energy);
  D.FinishGradient(grad, grad_ratio_bar, Energy);
}

template<typename Wfn, typename Walker> 
void getLanczosCoeffsDeterministic(Wfn &w, Walker &walk, double &alpha, Eigen::VectorXd &lanczosCoeffs)
{
  int norbs = Determinant::norbs;
  int nalpha = Determinant::nalpha;
  int nbeta = Determinant::nbeta;

  vector<Determinant> allDets;
  generateAllDeterminants(allDets, norbs, nalpha, nbeta);

  workingArray work, moreWork;

  double overlapTot = 0.; 
  Eigen::VectorXd coeffs = Eigen::VectorXd::Zero(4);
  //w.printVariables();

  for (int i = commrank; i < allDets.size(); i += commsize)
  {
    w.initWalker(walk, allDets[i]);
    Eigen::VectorXd coeffsSample = Eigen::VectorXd::Zero(4);
    double overlapSample = 0.;
    w.HamAndOvlpLanczos(walk, coeffsSample, overlapSample, work, moreWork, alpha);
    if (schd.debug) {
      cout << "walker\n" << walk << endl;
      cout << "coeffsSample\n" << coeffsSample << endl;
    }
    //cout << "ham  " << ham[0] << "  " << ham[1] << "  " << ham[2] << endl;
    //cout << "ovlp  " << ovlp[0] << "  " << ovlp[1] << "  " << ovlp[2] << endl << endl;
    
    //grad += localgrad * ovlp * ovlp;
    overlapTot += overlapSample * overlapSample;
    coeffs += (overlapSample * overlapSample) * coeffsSample;
  }

#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &(overlapTot), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, coeffs.data(), 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  lanczosCoeffs = coeffs / overlapTot;
}

//############################################################Continuous Time Evaluation############################################################################
template<typename Wfn, typename Walker> 
void getStochasticEnergyContinuousTime(Wfn &w, Walker &walk, double &Energy, double &stddev, double &rk, int niter)
{
  ContinuousTime<Wfn, Walker> CTMC(w, walk, niter);
  Energy = 0.0, stddev = 0.0, rk = 0.0;
  CTMC.LocalEnergy();
  for (int iter = 0; iter < niter; iter++)
  {
    CTMC.MakeMove();
    CTMC.UpdateEnergy(Energy);
    //CTMC.LocalEnergy();
    CTMC.LocalEnergy();
    //CTMC.UpdateBestDet();
    //CTMC.UpdateEnergy(Energy);
  }
  CTMC.FinishEnergy(Energy, stddev, rk);
  //CTMC.FinishBestDet();
}
    
template<typename Wfn, typename Walker>
void getStochasticGradientContinuousTime(Wfn &w, Walker &walk, double &Energy, double &stddev, VectorXd &grad, double &rk, int niter)
{
  ContinuousTime<Wfn, Walker> CTMC(w, walk, niter);
  Energy = 0.0, stddev = 0.0, rk = 0.0;
  grad.setZero();
  VectorXd grad_ratio_bar = VectorXd::Zero(grad.rows());
  CTMC.LocalEnergy();
  CTMC.LocalGradient();
  for (int iter = 0; iter < niter; iter++)
  {
    CTMC.MakeMove();
    CTMC.UpdateEnergy(Energy);
    CTMC.UpdateGradient(grad, grad_ratio_bar);
    //CTMC.LocalEnergy();
    //CTMC.LocalGradient();
    CTMC.LocalEnergy();
    CTMC.LocalGradient();
    CTMC.UpdateBestDet();
    //CTMC.UpdateEnergy(Energy);
    //CTMC.UpdateGradient(grad, grad_ratio_bar);
  }
  CTMC.FinishEnergy(Energy, stddev, rk);
  CTMC.FinishGradient(grad, grad_ratio_bar, Energy);
  CTMC.FinishBestDet();
}
    
template<typename Wfn, typename Walker>
void getStochasticGradientMetricContinuousTime(Wfn &w, Walker& walk, double &Energy, double &stddev, VectorXd &grad, VectorXd &H, DirectMetric &S, double &rk, int niter)
{
  ContinuousTime<Wfn, Walker> CTMC(w, walk, niter);
  Energy = 0.0, stddev = 0.0, rk = 0.0;
  grad.setZero();
  VectorXd grad_ratio_bar = VectorXd::Zero(grad.rows());
  for (int iter = 0; iter < niter; iter++)
  {
    CTMC.LocalEnergy();
    CTMC.LocalGradient();
    CTMC.MakeMove();
    CTMC.UpdateEnergy(Energy);
    CTMC.UpdateGradient(grad, grad_ratio_bar);
    CTMC.UpdateSR(S);
    CTMC.UpdateBestDet();
  }
  CTMC.FinishEnergy(Energy, stddev, rk);
  CTMC.FinishGradient(grad, grad_ratio_bar, Energy);
  CTMC.FinishSR(grad, grad_ratio_bar, H, S);
  CTMC.FinishBestDet();
}

/*
template<typename Wfn, typename Walker>
void getStochasticGradientVarianceContinuousTime(Wfn &w, Walker &walk, double &Variance, double &Energy, double &stddev, VectorXd &grad, DirectVarLM &H, double &rk, int niter)
{
  ContinuousTime<Wfn, Walker> CTMC(w, walk, niter);
  Energy = 0.0, Variance = 0.0, stddev = 0.0;
  grad.setZero();
  VectorXd grad_ratio_bar = VectorXd::Zero(grad.rows());
  for (int iter = 0; iter < niter; iter++)
  {
    CTMC.LocalEnergy();
    CTMC.LocalGradient();
    CTMC.LocalEnergyGradient(rk); 
    CTMC.MakeMove();
    CTMC.UpdateEnergy(Energy);
    CTMC.UpdateGradient(grad, grad_ratio_bar);
    CTMC.UpdateVariance(Variance, H, rk);
    CTMC.UpdateBestDet();
  }
  CTMC.FinishEnergy(Energy, stddev, rk);
  CTMC.FinishGradient(grad, grad_ratio_bar, Energy);
  CTMC.FinishVariance(Variance, Energy);
  CTMC.FinishBestDet();
}
*/

template<typename Wfn, typename Walker> 
void getStochasticOneRdmContinuousTime(Wfn &w, Walker &walk, MatrixXd &oneRdm, bool sz, int niter)
{
  int norbs = Determinant::norbs;
  int nalpha = Determinant::nalpha;
  int nbeta = Determinant::nbeta;

  auto random = std::bind(std::uniform_real_distribution<double>(0, 1),
                          std::ref(generator));

  workingArray work;
  int iter = 0;
  double ovlp = 0.;

  oneRdm = MatrixXd::Constant(norbs, norbs, 0.); 
  MatrixXd localOneRdm = MatrixXd::Constant(norbs, norbs, 0.); 
  double bestOvlp = 0.;

  double cumdeltaT = 0., cumdeltaT2 = 0.;
  w.initWalker(walk);
  Determinant bestDet = walk.d;

  while (iter < niter) {
    work.setCounterToZero();
    generateAllScreenedSingleExcitation(walk.d, schd.epsilon, schd.screen,
                                        work, false);  
    generateAllScreenedDoubleExcitation(walk.d, schd.epsilon, schd.screen,
                                        work, false);  
    double cumovlpRatio = 0;
    for (int i = 0; i < work.nExcitations; i++) {
      int ex1 = work.excitation1[i], ex2 = work.excitation2[i];
      double tia = work.HijElement[i];
      int I = ex1 / 2 / norbs, A = ex1 - 2 * norbs * I;
      int J = ex2 / 2 / norbs, B = ex2 - 2 * norbs * J;
      cumovlpRatio += abs(w.getOverlapFactor(I, J, A, B, walk, false));
      work.ovlpRatio[i] = cumovlpRatio;
    }
    double deltaT = 1.0 / (cumovlpRatio);
    double nextDetRandom = random() * cumovlpRatio;
    int nextDet = std::lower_bound(work.ovlpRatio.begin(), (work.ovlpRatio.begin() + work.nExcitations), nextDetRandom) - work.ovlpRatio.begin();
    cumdeltaT += deltaT;
    double ratio = deltaT / cumdeltaT;
    
    localOneRdm.setZero(norbs, norbs);
    vector<int> open;
    vector<int> closed;
    walk.d.getOpenClosed(sz, open, closed);
    for (int p = 0; p < closed.size(); p++) {
      localOneRdm(closed[p], closed[p]) = 1.;
      for (int q = 0; q < open.size() && open[q] < closed[p]; q++) {
        localOneRdm(closed[p], open[q]) = w.getOverlapFactor(2*closed[p] + sz, 2*open[q] + sz, walk, 0);
      }
    }
    oneRdm += deltaT * (localOneRdm - oneRdm) / (cumdeltaT); //running average of energy

    iter++;
    walk.updateWalker(w.getRef(), w.getCorr(), work.excitation1[nextDet], work.excitation2[nextDet]);
    double ovlp = w.Overlap(walk);
    if (abs(ovlp) > bestOvlp)
    {
      bestOvlp = abs(ovlp);
      bestDet = walk.getDet();
    }
  }

#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, oneRdm.data(), norbs*norbs, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  oneRdm = oneRdm / commsize;
  if (commrank == 0)
  {
    char file[5000];
    sprintf(file, "BestDeterminant.txt");
    std::ofstream ofs(file, std::ios::binary);
    boost::archive::binary_oarchive save(ofs);
    save << bestDet;
  }
}


template<typename Wfn, typename Walker> 
void getStochasticTwoRdmContinuousTime(Wfn &w, Walker &walk, MatrixXd &twoRdm, int niter)
{
  int norbs = Determinant::norbs;
  int nalpha = Determinant::nalpha;
  int nbeta = Determinant::nbeta;
  int nelec = nalpha + nbeta;
  
  auto random = std::bind(std::uniform_real_distribution<double>(0, 1),
                          std::ref(generator));

  workingArray work;
  int iter = 0;
  double ovlp = 0.;

  twoRdm = MatrixXd::Constant(norbs* norbs, norbs* norbs, 0.0); 
  double bestOvlp = 0.;

  double cumdeltaT = 0., cumdeltaT2 = 0.;
  w.initWalker(walk);
  Determinant bestDet = walk.d;

  while (iter < niter) {
    work.setCounterToZero();
    generateAllScreenedSingleExcitation(walk.d, schd.epsilon, schd.screen,
                                        work, false);  
    generateAllScreenedDoubleExcitation(walk.d, schd.epsilon, schd.screen,
                                        work, false);  
    double cumovlpRatio = 0;
    for (int i = 0; i < work.nExcitations; i++) {
      int ex1 = work.excitation1[i], ex2 = work.excitation2[i];
      double tia = work.HijElement[i];
      int I = ex1 / 2 / norbs, A = ex1 - 2 * norbs * I;
      int J = ex2 / 2 / norbs, B = ex2 - 2 * norbs * J;
      cumovlpRatio += abs(w.getOverlapFactor(I, J, A, B, walk, false));
      work.ovlpRatio[i] = cumovlpRatio;
    }
    double deltaT = 1.0 / (cumovlpRatio);
    double nextDetRandom = random() * cumovlpRatio;
    int nextDet = std::lower_bound(work.ovlpRatio.begin(), (work.ovlpRatio.begin() + work.nExcitations), nextDetRandom) - work.ovlpRatio.begin();
    cumdeltaT += deltaT;
    double ratio = deltaT / cumdeltaT;

    vector<int> open;
    vector<int> closed;
    walk.d.getOpenClosed(open, closed);
    
    twoRdm *= (1. - deltaT/cumdeltaT);

    double contribution = deltaT/cumdeltaT;
    
    //diagonal elements
    for (int a = 0; a < closed.size(); a++) 
      for (int b = 0; b < closed.size(); b++)  {
        if (a == b) continue;
        int A = closed[a]/2, B = closed[b]/2;
        twoRdm(A*norbs + B, B*norbs +A) += contribution * 0.5;
        twoRdm(B*norbs + A, A*norbs +B) += contribution * 0.5;
        if (closed[a]%2 == closed[b]%2) {
          twoRdm(A*norbs + B, A*norbs +B) += -contribution * 0.5;
          twoRdm(B*norbs + A, B*norbs +A) += -contribution * 0.5;
        }
      }


    //singleExcitations contribution
    for (int a = 0; a < closed.size(); a++) 
      for (int i = 0; i < open.size(); i++) {
        if (closed[a]%2 != open[i]%2) continue; 
        double ovlpratio = w.getOverlapFactor(closed[a], open[i], walk, 0);
        for (int b = 0; b < closed.size(); b++) {          
          if (a == b) continue;          
          updateTwoRdm(closed[a], closed[b], closed[b], open[i], norbs, contribution*ovlpratio, twoRdm);          
        }
    }

    
    //twordm contribution
    for (int a = 0; a < closed.size(); a++) 
      for (int b = a+1; b < closed.size() ; b++)  {
        for (int i = 0; i < open.size(); i++) 
          for (int j = i+1; j < open.size() ; j++) {
            if (closed[a]%2+closed[b]%2 - open[i]%2-open[j]%2 == 0) {
              int A = closed[a], B = closed[b], I = open[i], J = open[j];
              if (A%2 != I%2) {J = open[i]; I = open[j];}
              
              double ovlpratio = w.getOverlapFactor(A, B, I, J, walk, 0);
              updateTwoRdm(A, B, J, I, norbs, contribution*ovlpratio, twoRdm);
            }
          }
      }
    
    
    
    iter++;
    walk.updateWalker(w.getRef(), w.getCorr(), work.excitation1[nextDet], work.excitation2[nextDet]);
    double ovlp = w.Overlap(walk);
    if (abs(ovlp) > bestOvlp)
    {
      bestOvlp = abs(ovlp);
      bestDet = walk.getDet();
    }
  }
  
#ifndef SERIAL
  size_t size = norbs*norbs*norbs*norbs;
  MPI_Allreduce(MPI_IN_PLACE, twoRdm.data(), size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
  
  twoRdm = twoRdm / commsize;
  if (commrank == 0)
  {
    char file[5000];
    sprintf(file, "BestDeterminant.txt");
    std::ofstream ofs(file, std::ios::binary);
    boost::archive::binary_oarchive save(ofs);
    save << bestDet;
  }

  if (commrank == 0) {
    char file[5000];
    int root = 0;
    sprintf(file, "spatialRDM.%d.%d.txt", root,
            root);
    std::ofstream ofs(file, std::ios::out);
    ofs << norbs << endl;
    double Etwo = 0.0, Eone = 0.0;
    for (int i=0; i<norbs; i++)
      for (int j=0; j<norbs; j++)
        for (int a=0; a<norbs; a++)
          for (int b=0; b<norbs; b++) {
            ofs << str(boost::format("%3d   %3d   %3d   %3d   %10.8g\n") %
                       i % j % b % a %
                       (twoRdm(i * norbs + j, a * norbs + b)));
            //double int2 = I2 (2*i, 2*b, 2*j, 2*a);
            //Etwo += 0.5 * twoRdm(i*norbs + j , a*norbs + b) * int2;
          }
    ofs.close();
    
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

template<typename Wfn, typename Walker> 
void getStochasticDensityCorrelationsContinuousTime(Wfn &w, Walker &walk, MatrixXd &corr, int niter)
{
  int norbs = Determinant::norbs;
  int nalpha = Determinant::nalpha;
  int nbeta = Determinant::nbeta;

  auto random = std::bind(std::uniform_real_distribution<double>(0, 1),
                          std::ref(generator));

  workingArray work;
  int iter = 0;
  double ovlp = 0.;

  corr = MatrixXd::Constant(2*norbs, 2*norbs, 0.);
  MatrixXd localCorr = MatrixXd::Constant(2*norbs, 2*norbs, 0.); 
  double bestOvlp = 0.;

  double cumdeltaT = 0., cumdeltaT2 = 0.;
  w.initWalker(walk);
  Determinant bestDet = walk.d;

  while (iter < niter) {
    work.setCounterToZero();
    generateAllScreenedSingleExcitation(walk.d, schd.epsilon, schd.screen,
                                        work, false);  
    generateAllScreenedDoubleExcitation(walk.d, schd.epsilon, schd.screen,
                                        work, false);  
    double cumovlpRatio = 0;
    for (int i = 0; i < work.nExcitations; i++) {
      int ex1 = work.excitation1[i], ex2 = work.excitation2[i];
      double tia = work.HijElement[i];
      int I = ex1 / 2 / norbs, A = ex1 - 2 * norbs * I;
      int J = ex2 / 2 / norbs, B = ex2 - 2 * norbs * J;
      cumovlpRatio += abs(w.getOverlapFactor(I, J, A, B, walk, false));
      work.ovlpRatio[i] = cumovlpRatio;
    }
    double deltaT = 1.0 / (cumovlpRatio);
    double nextDetRandom = random() * cumovlpRatio;
    int nextDet = std::lower_bound(work.ovlpRatio.begin(), (work.ovlpRatio.begin() + work.nExcitations), nextDetRandom) - work.ovlpRatio.begin();
    cumdeltaT += deltaT;
    double ratio = deltaT / cumdeltaT;
    
    localCorr.setZero(2*norbs, 2*norbs);
    vector<int> open;
    vector<int> closed;
    walk.d.getOpenClosed(open, closed);
    for (int p = 0; p < closed.size(); p++) {
      localCorr(closed[p], closed[p]) = 1.;
      for (int q = 0; q < p; q++) {
        int P = max(closed[p], closed[q]), Q = min(closed[p], closed[q]);
        localCorr(P, Q) = 1.;
      }
    }
    corr += deltaT * (localCorr - corr) / (cumdeltaT); //running average of energy

    iter++;
    walk.updateWalker(w.getRef(), w.getCorr(), work.excitation1[nextDet], work.excitation2[nextDet]);
    double ovlp = w.Overlap(walk);
    if (abs(ovlp) > bestOvlp)
    {
      bestOvlp = abs(ovlp);
      bestDet = walk.getDet();
    }
  }

#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, corr.data(), 4*norbs*norbs, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  corr = corr / commsize;
  if (commrank == 0)
  {
    char file[5000];
    sprintf(file, "BestDeterminant.txt");
    std::ofstream ofs(file, std::ios::binary);
    boost::archive::binary_oarchive save(ofs);
    save << bestDet;
  }
}

template<typename Wfn, typename Walker> 
void getLanczosCoeffsContinuousTime(Wfn &w, Walker &walk, double &alpha, Eigen::VectorXd &lanczosCoeffs, Eigen::VectorXd &stddev,
                                       Eigen::VectorXd &rk, int niter, double targetError)
{
  int norbs = Determinant::norbs;
  int nalpha = Determinant::nalpha;
  int nbeta = Determinant::nbeta;

  auto random = std::bind(std::uniform_real_distribution<double>(0, 1),
                          std::ref(generator));

  int iter = 0;
  Eigen::VectorXd S1 = Eigen::VectorXd::Zero(4);
  Eigen::VectorXd coeffs = Eigen::VectorXd::Zero(4);
  Eigen::VectorXd coeffsSample = Eigen::VectorXd::Zero(4);
  double ovlpSample = 0.;

  double bestOvlp = 0.;
  Determinant bestDet = walk.getDet();

  workingArray work, moreWork;
  w.HamAndOvlpLanczos(walk, coeffsSample, ovlpSample, work, moreWork, alpha);

  int nstore = 1000000 / commsize;
  int gradIter = min(nstore, niter);

  std::vector<std::vector<double>> gradError;
  gradError.resize(4);
  //std::vector<double> gradError(gradIter * commsize, 0.);
  vector<double> tauError(gradIter * commsize, 0.);
  for (int i = 0; i < 4; i++)
    gradError[i] = std::vector<double>(gradIter * commsize, 0.);
  double cumdeltaT = 0.;
  double cumdeltaT2 = 0.;
  
  int transIter = 0, nTransIter = niter/2;

  //while (transIter < nTransIter) {
  //  double cumovlpRatio = 0;
  //  //when using uniform probability 1./numConnection * max(1, pi/pj)
  //  for (int i = 0; i < work.nExcitations; i++) {
  //    cumovlpRatio += abs(work.ovlpRatio[i]);
  //    work.ovlpRatio[i] = cumovlpRatio;
  //  }

  //  double nextDetRandom = random() * cumovlpRatio;
  //  int nextDet = std::lower_bound(work.ovlpRatio.begin(), (work.ovlpRatio.begin() + work.nExcitations),
  //                                 nextDetRandom) - work.ovlpRatio.begin();

  //  transIter++;
  //  walk.updateWalker(w.getRef(), w.getCorr(), work.excitation1[nextDet], work.excitation2[nextDet]);
  //  w.HamAndOvlpLanczos(walk, coeffsSample, ovlpSample, work, moreWork, alpha);
  //}

  while (iter < niter) {
    double cumovlpRatio = 0;
    //when using uniform probability 1./numConnection * max(1, pi/pj)
    for (int i = 0; i < work.nExcitations; i++) {
      cumovlpRatio += abs(work.ovlpRatio[i]);
      work.ovlpRatio[i] = cumovlpRatio;
    }

    //double deltaT = -log(random())/(cumovlpRatio);
    double deltaT = 1.0 / (cumovlpRatio);
    double nextDetRandom = random() * cumovlpRatio;
    int nextDet = std::lower_bound(work.ovlpRatio.begin(), (work.ovlpRatio.begin() + work.nExcitations),
                                   nextDetRandom) - work.ovlpRatio.begin();

    cumdeltaT += deltaT;
    cumdeltaT2 += deltaT * deltaT;
    double ratio = deltaT / cumdeltaT;
    Eigen::VectorXd coeffsOld = coeffs;
    coeffs += ratio * (coeffsSample - coeffs);
    S1 += deltaT * (coeffsSample - coeffsOld).cwiseProduct(coeffsSample - coeffs);

    if (iter < gradIter) {
      tauError[iter + commrank * gradIter] = deltaT;
      for (int i = 0; i < 4; i++) 
        gradError[i][iter + commrank * gradIter] = coeffsSample[i];
    }

    iter++;

    walk.updateWalker(w.getRef(), w.getCorr(), work.excitation1[nextDet], work.excitation2[nextDet]);
    w.HamAndOvlpLanczos(walk, coeffsSample, ovlpSample, work, moreWork, alpha);
  }
  

#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &(gradError[0][0]), gradError[0].size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(gradError[1][0]), gradError[1].size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(gradError[2][0]), gradError[2].size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(gradError[3][0]), gradError[3].size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(tauError[0]), tauError.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, coeffs.data(), 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  for (int i = 0; i < 4; i++)
  {
    vector<double> b_size, r_x;
    block(b_size, r_x, gradError[i], tauError);
    rk[i] = corrTime(gradError.size(), b_size, r_x);
    S1[i] /= cumdeltaT;
  }

  double n_eff = commsize * (cumdeltaT * cumdeltaT) / cumdeltaT2;
  for (int i = 0; i < 4; i++) { 
    stddev[i] = sqrt(S1[i] * rk[i] / n_eff);
  }

  lanczosCoeffs = coeffs / commsize;

}


template<typename Wfn, typename Walker>
double getEnergyMetropolisRealSpace(Wfn &wave, Walker &walk, double &E0, double &stddev, double &rk, int niter, double targetError)
{
  auto random = std::bind(std::uniform_real_distribution<double>(0, 1), std::ref(generator));

  rDeterminant bestDet = walk.getDet();
  double bestOvlp = std::pow(wave.Overlap(walk), 2);
 
  int nelec = walk.d.nelec;
  Statistics Stats; 
  double energy = 0.0, S1 = 0.0;

  double acceptedFrac = 0;
  int nstore = 1000000;
  int corrIter = std::min(nstore, niter);
  int effIter = 0;
  for (int iter = 0; iter < niter; iter++)
  {
    //make n-electron move
    for (int elec = 0; elec < nelec; elec++)
    {
      Vector3d step;
      double ovlpProb = 0.0, proposalProb = 0.0;
      walk.getStep(step, elec, schd.rStepSize, wave.getRef(), wave.getCorr(), ovlpProb, proposalProb);
      step += walk.d.coord[elec];  
      //if move gaussian or simple, ovlpProb not calculated
      if (ovlpProb < -0.5) ovlpProb = std::pow(wave.getOverlapFactor(elec, step, walk), 2); 
      
      //accept or reject move based on metropolis
      if (ovlpProb * proposalProb > random())
      {
        acceptedFrac++;
        walk.updateWalker(elec, step, wave.getRef(), wave.getCorr());
        double ovlp = std::pow(wave.Overlap(walk), 2);
        if (ovlp > bestOvlp)
        {
          bestOvlp = ovlp;
          bestDet = walk.getDet();
        }
      }
    }

    //sample energy and gradient
    if (iter > 0.01*niter)
    {
      effIter++;

      double eloc = wave.rHam(walk);

      if (schd.debug) cout << "eloc  " << eloc << endl;
      if (schd.debug) cout << "walker\n" << walk << endl;

      double oldEnergy = energy;
      energy += (eloc - energy) / effIter;
      S1 += (eloc - oldEnergy) * (eloc - energy);

      if (effIter < corrIter) Stats.push_back(eloc);
    }
  }

  try 
  {
    Stats.Block();
    rk = Stats.BlockCorrTime();
  }
  catch (const runtime_error &error)
  {
    Stats.CorrFunc();
    rk = Stats.IntCorrTime();
  }
  S1 /= effIter;
#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &S1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &rk, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &effIter, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  energy /= commsize;
  S1 /= commsize;
  rk /= commsize;
#endif
  double n_eff = effIter;
  stddev = std::sqrt(S1 * rk / n_eff);
  E0 = energy;

  if (commrank == 0)
  {
    char file[5000];
    sprintf(file, "BestCoordinates.bkp");
    std::ofstream ofs(file, std::ios::binary);
    boost::archive::binary_oarchive save(ofs);
    save << bestDet;
  } 
  return acceptedFrac / (niter * nelec);
}


template<typename Wfn, typename Walker>
double getGradientMetropolisRealSpace(Wfn &wave, Walker &walk, double &E0, double &stddev, Eigen::VectorXd &grad, double &rk, int niter, double targetError)
{
  ///////////////////////////////////////////////
  //Below will test overlap ratios from all the Jastrow functions
  /*
  cout << endl;
  cout << "Testing overlap factor" << endl;
  Walker walk1, walk2;
  wave.initWalker(walk1), wave.initWalker(walk2);
  cout << "#############Before move################" << endl;
  cout << "walk1" << endl;
  cout << walk1.d << endl;
  cout << walk1.corrHelper.exponential << endl;
  cout << endl;
  cout << "N" << endl;
  cout << walk1.corrHelper.N.transpose() << endl;
  cout << endl;
  cout << "n" << endl;
  cout << walk1.corrHelper.n << endl;
  cout << endl;
  cout << "gradn[0]" << endl;
  cout << walk2.corrHelper.gradn[0] << endl;
  cout << "gradn[1]" << endl;
  cout << walk2.corrHelper.gradn[1] << endl;
  cout << "gradn[2]" << endl;
  cout << walk2.corrHelper.gradn[2] << endl;
  cout << endl;
  cout << "lapn" << endl;
  cout << walk1.corrHelper.lapn << endl;
  cout << endl;
  cout << "grad ratio" << endl;
  cout << walk1.corrHelper.GradRatio << endl << endl;
  cout << "ParamValues" << endl;
  cout << walk1.corrHelper.ParamValues.transpose() << endl;
  cout << endl;

  cout << "walk2" << endl;
  cout << walk2.d << endl;
  cout << walk2.corrHelper.exponential << endl << endl;
  
  cout << "#############After move################" << endl;
  cout << "update" << endl;
  Vector3d move(0.0, 1.0, 0.0);
  move += walk1.d.coord[0];
  
  walk2.updateWalker(0, move, wave.getRef(), wave.getCorr());
  
  cout << "init" << endl;
  rDeterminant pd(walk1.d);
  pd.coord[0] = move;
  Walker walk3(wave.getCorr(), wave.getRef(), pd);
  
  cout << endl << "__Init__" << endl;
  cout << walk3.d << endl;
  cout << walk3.corrHelper.exponential << endl;
  cout << "diff: " << walk3.corrHelper.exponential - walk1.corrHelper.exponential << endl;
  cout << endl;
  cout << "N" << endl;
  cout << walk3.corrHelper.N.transpose() << endl;
  cout << endl;
  cout << "n" << endl;
  cout << walk3.corrHelper.n << endl;
  cout << endl;
  cout << "gradn[0]" << endl;
  cout << walk2.corrHelper.gradn[0] << endl;
  cout << "gradn[1]" << endl;
  cout << walk2.corrHelper.gradn[1] << endl;
  cout << "gradn[2]" << endl;
  cout << walk2.corrHelper.gradn[2] << endl;
  cout << endl;
  cout << "lapn" << endl;
  cout << walk3.corrHelper.lapn << endl;
  cout << endl;
  cout << "grad ratio" << endl;
  cout << walk3.corrHelper.GradRatio << endl << endl;
  cout << "ParamValues" << endl;
  cout << walk3.corrHelper.ParamValues.transpose() << endl;
  cout << endl;

  cout << endl << "__updateWalker__" << endl;
  cout << walk2.d << endl;
  cout << walk2.corrHelper.exponential << endl;
  cout << "diff: " << walk2.corrHelper.exponential - walk1.corrHelper.exponential << endl;
  cout << endl;
  cout << "N" << endl;
  cout << walk2.corrHelper.N.transpose() << endl;
  cout << endl;
  cout << "n" << endl;
  cout << walk2.corrHelper.n << endl;
  cout << endl;
  cout << "gradn[0]" << endl;
  cout << walk2.corrHelper.gradn[0] << endl;
  cout << "gradn[1]" << endl;
  cout << walk2.corrHelper.gradn[1] << endl;
  cout << "gradn[2]" << endl;
  cout << walk2.corrHelper.gradn[2] << endl;
  cout << endl;
  cout << "lapn" << endl;
  cout << walk2.corrHelper.lapn << endl;
  cout << endl;
  cout << "grad ratio" << endl;
  cout << walk2.corrHelper.GradRatio << endl << endl;
  cout << "ParamValues" << endl;
  cout << walk2.corrHelper.ParamValues.transpose() << endl;
  cout << endl;
  cout << "Delta ParamValues" << endl;
  cout << walk2.corrHelper.ParamValues.transpose() - walk1.corrHelper.ParamValues.transpose() << endl;
  cout << endl;

  cout << endl << "__GetGradientAndParamValues__" << endl;
  VectorXd pvec;
  cout << walk1.corrHelper.OverlapRatioAndParamGradient(0, move, wave.getCorr(), walk1.d, pvec) << endl;
  cout << pvec.transpose() << endl;

  cout << endl << "__OverlapFactorWalker__" << endl;
  double a =  wave.getOverlapFactor(0, move, walk1);
  cout << a << endl;
  cout << endl;

  cout << "__GetGradientAfterElectronMove__" << endl;
  Vector3d vect;
  double test = walk1.getGradientAfterSingleElectronMove(0, move, vect, wave.getRef());
  cout << test << endl;
  cout << vect.transpose() << endl;

  cout << endl << "__Symmetry__" << endl;
  rDeterminant d1(walk1.d);
  rDeterminant d2;
  d2.coord[0] = d1.coord[1];
  d2.coord[1] = d1.coord[0];
  Walker Walk1(wave.getCorr(), wave.getRef(), d1);
  Walker Walk2(wave.getCorr(), wave.getRef(), d2);
  cout << "walk 1: " << Walk1.corrHelper.exponential << endl;
  cout << "walk 2: " << Walk2.corrHelper.exponential << endl;
  cout << endl;
  cout << "##############################" << endl;
  ////////////////////////////////////////// 
  */
  auto random = std::bind(std::uniform_real_distribution<double>(0, 1), std::ref(generator));

  rDeterminant bestDet = walk.getDet();
  double bestOvlp = std::pow(wave.Overlap(walk), 2);
 
  int nelec = walk.d.nelec;
  Statistics Stats; 
  double energy = 0.0, S1 = 0.0;
  grad.setZero();
  VectorXd diagonalGrad = VectorXd::Zero(grad.rows());

  double acceptedFrac = 0;
  int nstore = 1000000;
  int corrIter = std::min(nstore, niter);
  int effIter = 0;
  for (int iter = 0; iter < niter; iter++)
  {
    //make n-electron move
    for (int elec = 0; elec < nelec; elec++)
    {
      Vector3d step;
      double ovlpProb = 0.0, proposalProb = 0.0;
      walk.getStep(step, elec, schd.rStepSize, wave.getRef(), wave.getCorr(), ovlpProb, proposalProb);
      step += walk.d.coord[elec];  
      //if move gaussian or simple, ovlpProb not calculated
      if (ovlpProb < -0.5) ovlpProb = std::pow(wave.getOverlapFactor(elec, step, walk), 2); 
      
      //accept or reject move based on metropolis
      if (ovlpProb * proposalProb > random())
      {
        acceptedFrac++;
        walk.updateWalker(elec, step, wave.getRef(), wave.getCorr());
        double ovlp = std::pow(wave.Overlap(walk), 2);
        if (ovlp > bestOvlp)
        {
          bestOvlp = ovlp;
          bestDet = walk.getDet();
        }
      }
    }

    //sample energy and gradient
    if (iter > 0.01*niter)
    {
      effIter++;

      double eloc = wave.rHam(walk);
      double ovlp = 0.0;
      VectorXd localdiagonalGrad = VectorXd::Zero(grad.size());
      wave.OverlapWithGradient(walk, ovlp, localdiagonalGrad);

      if (schd.debug) cout << "eloc  " << eloc << endl;
      if (schd.debug) cout << "walker\n" << walk << endl;

      double oldEnergy = energy;
      energy += (eloc - energy) / effIter;
      S1 += (eloc - oldEnergy) * (eloc - energy);

      diagonalGrad += (localdiagonalGrad - diagonalGrad) / effIter;
      grad += (eloc * localdiagonalGrad - grad) / effIter;

      if (effIter < corrIter) Stats.push_back(eloc);
    }
  }

  try 
  {
    Stats.Block();
    rk = Stats.BlockCorrTime();
  }
  catch (const runtime_error &error)
  {
    Stats.CorrFunc();
    rk = Stats.IntCorrTime();
  }
  S1 /= effIter;
#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &(diagonalGrad[0]), grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(grad[0]), grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &S1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &rk, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &effIter, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  diagonalGrad /= commsize;
  grad /= commsize;
  energy /= commsize;
  S1 /= commsize;
  rk /= commsize;
#endif
  double n_eff = effIter;
  stddev = std::sqrt(S1 * rk / n_eff);
  E0 = energy;
  grad = grad - E0 * diagonalGrad;

  if (commrank == 0)
  {
    char file[5000];
    sprintf(file, "BestCoordinates.bkp");
    std::ofstream ofs(file, std::ios::binary);
    boost::archive::binary_oarchive save(ofs);
    save << bestDet;
  } 
  return acceptedFrac / (niter * nelec);
}

template<typename Wfn, typename Walker>
double getGradientMetricMetropolisRealSpace(Wfn &wave, Walker &walk, double &E0, double &stddev, Eigen::VectorXd &grad, VectorXd& H, DirectMetric& S, double &rk, int niter, double targetError)
{
  auto random = std::bind(std::uniform_real_distribution<double>(0, 1), std::ref(generator));

  rDeterminant bestDet = walk.getDet();
  double bestOvlp = std::pow(wave.Overlap(walk), 2);
 
  int nelec = walk.d.nelec;
  Statistics Stats; 
  double energy = 0.0, S1 = 0.0;
  int numVars = grad.rows();
  grad.setZero();
  H = VectorXd::Zero(numVars + 1);
  VectorXd diagonalGrad = VectorXd::Zero(numVars);

  double acceptedFrac = 0;
  int nstore = 1000000;
  int corrIter = std::min(nstore, niter);
  int effIter = 0;
  for (int iter = 0; iter < niter; iter++)
  {
    //make n-electron move
    for (int elec = 0; elec < nelec; elec++)
    {
      Vector3d step;
      double ovlpProb = 0.0, proposalProb = 0.0;
      walk.getStep(step, elec, schd.rStepSize, wave.getRef(), wave.getCorr(), ovlpProb, proposalProb);
      step += walk.d.coord[elec];  
      //if move gaussian or simple, ovlpProb not calculated
      if (ovlpProb < -0.5) ovlpProb = std::pow(wave.getOverlapFactor(elec, step, walk), 2); 
      
      //accept or reject move based on metropolis
      if (ovlpProb * proposalProb > random())
      {
        acceptedFrac++;
        walk.updateWalker(elec, step, wave.getRef(), wave.getCorr());
        double ovlp = std::pow(wave.Overlap(walk), 2);
        if (ovlp > bestOvlp)
        {
          bestOvlp = ovlp;
          bestDet = walk.getDet();
        }
      }
    }

    //sample energy and gradient
    if (iter > 0.01*niter)
    {
      effIter++;

      double eloc = wave.rHam(walk);
      double ovlp = 0.0;
      VectorXd localdiagonalGrad = VectorXd::Zero(numVars);
      wave.OverlapWithGradient(walk, ovlp, localdiagonalGrad);

      if (schd.debug) cout << "eloc  " << eloc << endl;
      if (schd.debug) cout << "walker\n" << walk << endl;

      double oldEnergy = energy;
      energy += (eloc - energy) / effIter;
      S1 += (eloc - oldEnergy) * (eloc - energy);

      diagonalGrad += (localdiagonalGrad - diagonalGrad) / effIter;
      grad += (eloc * localdiagonalGrad - grad) / effIter;

      VectorXd appended;
      appended << 1.0, localdiagonalGrad;
      if (schd.direct) {
        S.Vectors.push_back(appended);
        S.T.push_back(1.0);
      }
      else {
        S.Smatrix.noalias() += (appended * appended.transpose() - S.Smatrix) / effIter;
      }

      if (effIter < corrIter) Stats.push_back(eloc);
    }
  }

  try 
  {
    Stats.Block();
    rk = Stats.BlockCorrTime();
  }
  catch (const runtime_error &error)
  {
    Stats.CorrFunc();
    rk = Stats.IntCorrTime();
  }
  S1 /= effIter;
#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &S1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &rk, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &effIter, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(diagonalGrad[0]), grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(grad[0]), grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  if (!schd.direct) { MPI_Allreduce(MPI_IN_PLACE, S.Smatrix.data(), S.Smatrix.rows() * S.Smatrix.cols(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); }

  energy /= commsize;
  S1 /= commsize;
  rk /= commsize;
  diagonalGrad /= commsize;
  grad /= commsize;
  if (!schd.direct) { S.Smatrix /= commsize; }
#endif
  double n_eff = effIter;
  stddev = std::sqrt(S1 * rk / n_eff);
  E0 = energy;

  grad = grad - E0 * diagonalGrad;
  H << 1.0, (diagonalGrad - schd.stepsize * grad);

  if (commrank == 0)
  {
    char file[5000];
    sprintf(file, "BestCoordinates.bkp");
    std::ofstream ofs(file, std::ios::binary);
    boost::archive::binary_oarchive save(ofs);
    save << bestDet;
  } 
  return acceptedFrac / (niter * nelec);
}


template<typename Wfn, typename Walker>
double getGradientHessianMetropolisRealSpace(Wfn &wave, Walker &walk, double &E0, double &stddev, Eigen::VectorXd &grad, MatrixXd& Hessian, MatrixXd& Smatrix, double &rk, int niter)
{
  auto random = std::bind(std::uniform_real_distribution<double>(0, 1), std::ref(generator));

  rDeterminant bestDet = walk.getDet();
  double bestOvlp = std::pow(wave.Overlap(walk), 2);
 
  int nelec = walk.d.nelec;
  Statistics Stats; 
  double energy = 0.0, S1 = 0.0;
  int numVars = grad.rows();
  grad.setZero();
  Hessian = MatrixXd::Zero(numVars + 1, numVars + 1);
  Smatrix = MatrixXd::Zero(numVars + 1, numVars + 1);
  VectorXd gradRatio_bar = VectorXd::Zero(numVars);

  double acceptedFrac = 0;
  int nstore = 1000000;
  int corrIter = std::min(nstore, niter);
  int effIter = 0, eIter = 0;
  for (int iter = 0; iter < niter; iter++)
  {
    //make n-electron move
    for (int elec = 0; elec < nelec; elec++)
    {
      Vector3d step;
      double ovlpProb = 0.0, proposalProb = 0.0;
      walk.getStep(step, elec, schd.rStepSize, wave.getRef(), wave.getCorr(), ovlpProb, proposalProb);
      step += walk.d.coord[elec];  
      //if move gaussian or simple, ovlpProb not calculated
      if (ovlpProb < -0.5) ovlpProb = std::pow(wave.getOverlapFactor(elec, step, walk), 2); 
      
      //accept or reject move based on metropolis
      if (ovlpProb * proposalProb > random())
      {
        acceptedFrac++;
        walk.updateWalker(elec, step, wave.getRef(), wave.getCorr());
        double ovlp = std::pow(wave.Overlap(walk), 2);
        if (ovlp > bestOvlp)
        {
          bestOvlp = ovlp;
          bestDet = walk.getDet();
        }
      }
    }

    //sample energy and gradient
    if (iter > 0.01*niter)
    {
      eIter++;

      double eloc = 0.0;
      if (iter % int(rk))
      {
        eloc = wave.rHam(walk);
      }
      else
      {
        effIter++;

        VectorXd gradRatio = VectorXd::Zero(numVars);
        VectorXd hamRatio = VectorXd::Zero(numVars);
        eloc = wave.HamOverlap(walk, gradRatio, hamRatio);

        VectorXd G(numVars + 1), H(numVars + 1);
        G << 1.0, gradRatio;
        H << eloc, hamRatio;

        Hessian.noalias() += (G * H.transpose() - Hessian) / effIter;
        Smatrix.noalias() += (G * G.transpose() - Smatrix) / effIter;

        gradRatio_bar += (gradRatio - gradRatio_bar) / effIter;
        grad += (eloc * gradRatio - grad) / effIter;
      }

      double oldEnergy = energy;
      energy += (eloc - energy) / eIter;
      S1 += (eloc - oldEnergy) * (eloc - energy);

      if (eIter < corrIter) Stats.push_back(eloc);

      if (schd.debug) cout << "eloc  " << eloc << endl;
      if (schd.debug) cout << "walker\n" << walk << endl;
    }
  }

  try 
  {
    Stats.Block();
    rk = Stats.BlockCorrTime();
  }
  catch (const runtime_error &error)
  {
    Stats.CorrFunc();
    rk = Stats.IntCorrTime();
  }
  S1 /= eIter;
#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &(Hessian(0,0)), Hessian.rows()*Hessian.cols(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(Smatrix(0,0)), Smatrix.rows()*Smatrix.cols(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(gradRatio_bar[0]), grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(grad[0]), grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &S1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &rk, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &eIter, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  gradRatio_bar /= commsize;
  grad /= commsize;
  Hessian /= commsize;
  Smatrix /= commsize;
  energy /= commsize;
  S1 /= commsize;
  rk /= commsize;
#endif
  double n_eff = eIter;
  stddev = std::sqrt(S1 * rk / n_eff);
  E0 = energy;

  grad = grad - E0 * gradRatio_bar;

  if (commrank == 0)
  {
    char file[5000];
    sprintf(file, "BestCoordinates.bkp");
    std::ofstream ofs(file, std::ios::binary);
    boost::archive::binary_oarchive save(ofs);
    save << bestDet;
  } 
  return acceptedFrac / (niter * nelec);
}

template<typename Wfn, typename Walker>
double getGradientHessianDirectMetropolisRealSpace(Wfn &wave, Walker &walk, double &E0, double &stddev, Eigen::VectorXd &grad, DirectLM &h, double &rk, int niter)
{
  auto random = std::bind(std::uniform_real_distribution<double>(0, 1), std::ref(generator));

  rDeterminant bestDet = walk.getDet();
  double bestOvlp = std::pow(wave.Overlap(walk), 2);
 
  int nelec = walk.d.nelec;
  Statistics Stats; 
  double energy = 0.0, S1 = 0.0;
  int numVars = grad.rows();
  grad.setZero();
  VectorXd gradRatio_bar = VectorXd::Zero(numVars);

  double acceptedFrac = 0;
  int nstore = 1000000;
  int corrIter = std::min(nstore, niter);
  int effIter = 0, eIter = 0;
  for (int iter = 0; iter < niter; iter++)
  {
    //make n-electron move
    for (int elec = 0; elec < nelec; elec++)
    {
      Vector3d step;
      double ovlpProb = 0.0, proposalProb = 0.0;
      walk.getStep(step, elec, schd.rStepSize, wave.getRef(), wave.getCorr(), ovlpProb, proposalProb);
      step += walk.d.coord[elec];  
      //if move gaussian or simple, ovlpProb not calculated
      if (ovlpProb < -0.5) ovlpProb = std::pow(wave.getOverlapFactor(elec, step, walk), 2); 
      
      //accept or reject move based on metropolis
      if (ovlpProb * proposalProb > random())
      {
        acceptedFrac++;
        walk.updateWalker(elec, step, wave.getRef(), wave.getCorr());
        double ovlp = std::pow(wave.Overlap(walk), 2);
        if (ovlp > bestOvlp)
        {
          bestOvlp = ovlp;
          bestDet = walk.getDet();
        }
      }
    }

    //sample energy and gradient
    if (iter > 0.01*niter)
    {
      eIter++;

      double eloc = 0.0;
      if (iter % int(rk))
      {
        eloc = wave.rHam(walk);
      }
      else
      {
        effIter++;

        VectorXd gradRatio = VectorXd::Zero(numVars);
        VectorXd hamRatio = VectorXd::Zero(numVars);
        eloc = wave.HamOverlap(walk, gradRatio, hamRatio);

        VectorXd G(numVars + 1), H(numVars + 1);
        G << 0.0, gradRatio;
        H << 0.0, hamRatio;

        h.H.push_back(H);
        h.G.push_back(G);
        h.T.push_back(1.0);
        h.Eloc.push_back(eloc);

        gradRatio_bar += (gradRatio - gradRatio_bar) / effIter;
        grad += (eloc * gradRatio - grad) / effIter;
      }

      double oldEnergy = energy;
      energy += (eloc - energy) / eIter;
      S1 += (eloc - oldEnergy) * (eloc - energy);

      if (eIter < corrIter) Stats.push_back(eloc);

      if (schd.debug) cout << "eloc  " << eloc << endl;
      if (schd.debug) cout << "walker\n" << walk << endl;
    }
  }

  try 
  {
    Stats.Block();
    rk = Stats.BlockCorrTime();
  }
  catch (const runtime_error &error)
  {
    Stats.CorrFunc();
    rk = Stats.IntCorrTime();
  }
  S1 /= eIter;
#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &(gradRatio_bar[0]), grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(grad[0]), grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &S1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &rk, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &eIter, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  gradRatio_bar /= commsize;
  grad /= commsize;
  energy /= commsize;
  S1 /= commsize;
  rk /= commsize;
#endif
  double n_eff = eIter;
  stddev = std::sqrt(S1 * rk / n_eff);
  E0 = energy;

  grad = grad - E0 * gradRatio_bar;

  if (commrank == 0)
  {
    char file[5000];
    sprintf(file, "BestCoordinates.bkp");
    std::ofstream ofs(file, std::ios::binary);
    boost::archive::binary_oarchive save(ofs);
    save << bestDet;
  } 
  return acceptedFrac / (niter * nelec);
}

template<typename Wfn, typename Walker>
void getStochasticGradientHessianContinuousTime(Wfn &w, Walker& walk, double &Energy, double &stddev, VectorXd &grad, MatrixXd& Hmatrix, MatrixXd &Smatrix, double &rk, int niter)
{
  ContinuousTime<Wfn, Walker> CTMC(w, walk, niter);
  int numVars = grad.rows();
  Energy = 0.0;
  grad.setZero();
  Eigen::VectorXd grad_ratio_bar = VectorXd::Zero(numVars);
  Hmatrix = MatrixXd::Zero(numVars + 1, numVars+1);
  Smatrix = MatrixXd::Zero(numVars + 1, numVars+1);
  for (int iter = 0; iter < niter; iter++)
  {
    CTMC.LocalEnergy();
    CTMC.LocalGradient(iter % (int) std::round(rk));
    CTMC.LocalEnergyGradient(iter % (int) std::round(rk));
    CTMC.MakeMove(iter % (int) std::round(rk));
    CTMC.UpdateEnergy(Energy);
    CTMC.UpdateGradient(grad, grad_ratio_bar, iter % (int) std::round(rk));
    CTMC.UpdateLM(Hmatrix, Smatrix, iter % (int) std::round(rk));
    CTMC.UpdateBestDet();
  }
  CTMC.FinishEnergy(Energy, stddev, rk);
  CTMC.FinishGradient(grad, grad_ratio_bar, Energy);
  CTMC.FinishLM(Hmatrix, Smatrix);
  CTMC.FinishBestDet();
}

template<typename Wfn, typename Walker>
void getStochasticGradientMetricRandomContinuousTime(Wfn &w, Walker& walk, double &Energy, double &stddev, VectorXd &grad, MatrixXd &O, MatrixXd &SO, double &rk, int niter)
{
  ContinuousTime<Wfn, Walker> CTMC(w, walk, niter);
  int numVars = grad.rows();
  Energy = 0.0;
  grad.setZero();
  Eigen::VectorXd grad_ratio_bar = VectorXd::Zero(numVars);
  SO = Eigen::MatrixXd::Zero(numVars + 1, O.cols());
  for (int iter = 0; iter < niter; iter++)
  {
    CTMC.LocalEnergy();
    CTMC.LocalGradient();
    CTMC.MakeMove();
    CTMC.UpdateEnergy(Energy);
    CTMC.UpdateGradient(grad, grad_ratio_bar);
    CTMC.UpdateSO(O, SO);
    CTMC.UpdateBestDet();
  }
  CTMC.FinishEnergy(Energy, stddev, rk);
  CTMC.FinishGradient(grad, grad_ratio_bar, Energy);
  CTMC.FinishSO(O, SO, grad_ratio_bar);
  CTMC.FinishBestDet();
}

template<typename Wfn, typename Walker>
double getGradientMetricRandomMetropolisRealSpace(Wfn &wave, Walker &walk, double &Energy, double &stddev, Eigen::VectorXd &grad, MatrixXd& O, MatrixXd& SO, double &rk, int niter)
{
  auto random = std::bind(std::uniform_real_distribution<double>(0, 1), std::ref(generator));

  rDeterminant bestDet = walk.getDet();
  double bestOvlp = std::pow(wave.Overlap(walk), 2);
 
  int nelec = walk.d.nelec;
  Statistics Stats; 
  double energy = 0.0, S1 = 0.0;
  int numVars = grad.rows();
  grad.setZero();
  Eigen::VectorXd gradRatio_bar = VectorXd::Zero(numVars);
  SO = Eigen::MatrixXd::Zero(numVars + 1, O.cols());

  double acceptedFrac = 0;
  int nstore = 1000000;
  int corrIter = std::min(nstore, niter);
  int effIter = 0, eIter = 0;
  for (int iter = 0; iter < niter; iter++)
  {
    //make n-electron move
    for (int elec = 0; elec < nelec; elec++)
    {
      Vector3d step;
      double ovlpProb = 0.0, proposalProb = 0.0;
      walk.getStep(step, elec, schd.rStepSize, wave.getRef(), wave.getCorr(), ovlpProb, proposalProb);
      step += walk.d.coord[elec];  
      //if move gaussian or simple, ovlpProb not calculated
      if (ovlpProb < -0.5) ovlpProb = std::pow(wave.getOverlapFactor(elec, step, walk), 2); 
      
      //accept or reject move based on metropolis
      if (ovlpProb * proposalProb > random())
      {
        acceptedFrac++;
        walk.updateWalker(elec, step, wave.getRef(), wave.getCorr());
        double ovlp = std::pow(wave.Overlap(walk), 2);
        if (ovlp > bestOvlp)
        {
          bestOvlp = ovlp;
          bestDet = walk.getDet();
        }
      }
    }

    //sample energy and gradient
    if (iter > 0.01*niter)
    {
      eIter++;

      double eloc = 0.0;
      if (iter % int(rk))
      {
        eloc = wave.rHam(walk);
      }
      else
      {
        effIter++;

        eloc = wave.rHam(walk);
        double ovlp = 0.0;
        VectorXd gradRatio = VectorXd::Zero(numVars);
        wave.OverlapWithGradient(walk, ovlp, gradRatio);

        VectorXd G(numVars + 1);
        G << 0.0, gradRatio;
        MatrixXd tempGTO = G.transpose() * O;
        SO.noalias() += (G * tempGTO - SO) / effIter;

        gradRatio_bar += (gradRatio - gradRatio_bar) / effIter;
        grad += (eloc * gradRatio - grad) / effIter;
      }

      double oldEnergy = energy;
      energy += (eloc - energy) / eIter;
      S1 += (eloc - oldEnergy) * (eloc - energy);

      if (eIter < corrIter) Stats.push_back(eloc);

      if (schd.debug) cout << "eloc  " << eloc << endl;
      if (schd.debug) cout << "walker\n" << walk << endl;
    }
  }

  try 
  {
    Stats.Block();
    rk = Stats.BlockCorrTime();
  }
  catch (const runtime_error &error)
  {
    Stats.CorrFunc();
    rk = Stats.IntCorrTime();
  }
  S1 /= eIter;
#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, (SO.data()), SO.rows() * SO.cols(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(gradRatio_bar[0]), grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(grad[0]), grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &S1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &rk, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &eIter, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  SO /= commsize;
  gradRatio_bar /= commsize;
  grad /= commsize;
  energy /= commsize;
  S1 /= commsize;
  rk /= commsize;
#endif
  double n_eff = eIter;
  stddev = std::sqrt(S1 * rk / n_eff);
  Energy = energy;

  grad = grad - Energy * gradRatio_bar;

  Eigen::VectorXd e0 = Eigen::VectorXd::Unit(numVars + 1, 0);  
  Eigen::VectorXd Gappended_bar(numVars + 1);
  Gappended_bar << 0.0, gradRatio_bar;
  SO.noalias() = e0 * e0.transpose() * O 
               + SO 
               - Gappended_bar * Gappended_bar.transpose() * O;

  if (commrank == 0)
  {
    char file[5000];
    sprintf(file, "BestCoordinates.bkp");
    std::ofstream ofs(file, std::ios::binary);
    boost::archive::binary_oarchive save(ofs);
    save << bestDet;
  } 
  return acceptedFrac / (niter * nelec);
}

template<typename Wfn, typename Walker>
void getStochasticGradientHessianRandomContinuousTime(Wfn &w, Walker& walk, double &Energy, double &stddev, VectorXd &grad, MatrixXd &Q, MatrixXd &HQ, MatrixXd &SQ, double &rk, int niter)
{
  ContinuousTime<Wfn, Walker> CTMC(w, walk, niter);
  int numVars = grad.rows();
  Energy = 0;
  grad.setZero();
  HQ = MatrixXd::Zero(numVars + 1, Q.cols());
  SQ = MatrixXd::Zero(numVars + 1, Q.cols());
  Eigen::VectorXd G = VectorXd::Zero(numVars);
  Eigen::VectorXd h = VectorXd::Zero(numVars);
  Eigen::VectorXd g = VectorXd::Zero(numVars);
  for (int iter = 0; iter < niter; iter++)
  {
    int sample = iter % (int) std::round(rk);
    CTMC.LocalEnergy();
    CTMC.LocalGradient(sample);
    CTMC.LocalEnergyGradient(sample);
    CTMC.MakeMove(sample);
    CTMC.UpdateEnergy(Energy, sample);
    CTMC.UpdateRowAndColGradient(G, g, h, sample);
    CTMC.UpdateSO(Q, SQ, sample);
    CTMC.UpdateHO(Q, HQ, sample);
    //CTMC.UpdateBestDet();
  }
  Eigen::VectorXd Gr(numVars);
  Eigen::VectorXd Gc(numVars);
  CTMC.FinishEnergy(Energy, stddev, rk, 0);
  CTMC.FinishRowAndColGradient(Gr, Gc, G, g, h, Energy);
  grad = Gc;
  CTMC.FinishSO(Q, SQ, g);
  CTMC.FinishHO(Q, HQ, g, h, G, Gr, Gc, Energy);
  //CTMC.FinishBestDet();
}


template<typename Wfn, typename Walker>
double getGradientHessianRandomMetropolisRealSpace(Wfn &wave, Walker &walk, double &Energy, double &stddev, Eigen::VectorXd &grad, MatrixXd& Q, MatrixXd &HQ, MatrixXd& SQ, double &rk, int niter)
{
  auto random = std::bind(std::uniform_real_distribution<double>(0, 1), std::ref(generator));

  rDeterminant bestDet = walk.getDet();
  double bestOvlp = std::pow(wave.Overlap(walk), 2);
 
  int nelec = walk.d.nelec;
  Statistics Stats; 
  double energy = 0.0, S1 = 0.0;
  int numVars = grad.rows();
  grad.setZero();
  HQ = MatrixXd::Zero(numVars + 1, Q.cols());
  SQ = MatrixXd::Zero(numVars + 1, Q.cols());
  Eigen::VectorXd G = VectorXd::Zero(numVars);
  Eigen::VectorXd h = VectorXd::Zero(numVars);
  Eigen::VectorXd g = VectorXd::Zero(numVars);

  double acceptedFrac = 0;
  int nstore = 1000000;
  int corrIter = std::min(nstore, niter);
  int effIter = 0, eIter = 0;
  for (int iter = 0; iter < niter; iter++)
  {
    //make n-electron move
    for (int elec = 0; elec < nelec; elec++)
    {
      Vector3d step;
      double ovlpProb = 0.0, proposalProb = 0.0;
      walk.getStep(step, elec, schd.rStepSize, wave.getRef(), wave.getCorr(), ovlpProb, proposalProb);
      step += walk.d.coord[elec];  
      //if move gaussian or simple, ovlpProb not calculated
      if (ovlpProb < -0.5) ovlpProb = std::pow(wave.getOverlapFactor(elec, step, walk), 2); 
      
      //accept or reject move based on metropolis
      if (ovlpProb * proposalProb > random())
      {
        acceptedFrac++;
        walk.updateWalker(elec, step, wave.getRef(), wave.getCorr());
        double ovlp = std::pow(wave.Overlap(walk), 2);
        if (ovlp > bestOvlp)
        {
          bestOvlp = ovlp;
          bestDet = walk.getDet();
        }
      }
    }

    //sample energy and gradient
    if (iter > 0.01*niter)
    {
      eIter++;

      double eloc = 0.0;
      if (iter % int(rk))
      {
        eloc = wave.rHam(walk);
      }
      else
      {
        effIter++;

        Eigen::VectorXd gradRatio = VectorXd::Zero(numVars);
        Eigen::VectorXd hamRatio = VectorXd::Zero(numVars);
        eloc = wave.HamOverlap(walk, gradRatio, hamRatio);

        g += (gradRatio - g) / effIter;
        G += (gradRatio * eloc - G) / effIter;
        h += (hamRatio - h) / effIter;

        VectorXd G(numVars + 1), H(numVars + 1);
        G << 0.0, gradRatio;
        H << 0.0, hamRatio;
        MatrixXd tempGTQ = G.transpose() * Q;
        MatrixXd tempHTQ = H.transpose() * Q;

        SQ.noalias() += (G * tempGTQ - SQ) / effIter;
        HQ.noalias() += (G * tempHTQ - HQ) / effIter;
      }

      double oldEnergy = energy;
      energy += (eloc - energy) / eIter;
      S1 += (eloc - oldEnergy) * (eloc - energy);

      if (eIter < corrIter) Stats.push_back(eloc);

      if (schd.debug) cout << "eloc  " << eloc << endl;
      if (schd.debug) cout << "walker\n" << walk << endl;
    }
  }

  try 
  {
    Stats.Block();
    rk = Stats.BlockCorrTime();
  }
  catch (const runtime_error &error)
  {
    Stats.CorrFunc();
    rk = Stats.IntCorrTime();
  }
  S1 /= eIter;
#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, (G.data()), G.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, (g.data()), g.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, (h.data()), h.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, (SQ.data()), SQ.rows() * SQ.cols(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, (HQ.data()), HQ.rows() * HQ.cols(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &S1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &rk, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &eIter, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  G /= commsize;
  g /= commsize;
  h /= commsize;
  SQ /= commsize;
  HQ /= commsize;
  energy /= commsize;
  S1 /= commsize;
  rk /= commsize;
#endif
  double n_eff = eIter;
  stddev = std::sqrt(S1 * rk / n_eff);
  Energy = energy;

  VectorXd Gr, Gc;
  Gc = (G - energy * g);
  Gr = (h - energy * g);
  grad = Gc;

  Eigen::VectorXd e0 = Eigen::VectorXd::Unit(numVars + 1, 0);  
  Eigen::VectorXd Gappended_bar(numVars + 1);
  Gappended_bar << 0.0, g;
  SQ.noalias() = e0 * e0.transpose() * Q 
               + SQ 
               - Gappended_bar * Gappended_bar.transpose() * Q;
  
  Eigen::VectorXd Happended_bar(numVars + 1);
  Eigen::VectorXd Gr_appended(numVars + 1);
  Eigen::VectorXd Gc_appended(numVars + 1);
  Eigen::VectorXd G_appended(numVars + 1);
  Happended_bar << 0.0, h;
  Gr_appended << 0.0, Gr;
  Gc_appended << 0.0, Gc;
  G_appended << 0.0, G;
  HQ.noalias() = Energy * e0 * e0.transpose() * Q 
               + e0 * Gr_appended.transpose() * Q
               + Gc_appended * e0.transpose() * Q
               + HQ
               - G_appended * Gappended_bar.transpose() * Q
               - Gappended_bar * Happended_bar.transpose() * Q
               + Energy * Gappended_bar * Gappended_bar.transpose() * Q;

  if (commrank == 0)
  {
    char file[5000];
    sprintf(file, "BestCoordinates.bkp");
    std::ofstream ofs(file, std::ios::binary);
    boost::archive::binary_oarchive save(ofs);
    save << bestDet;
  } 
  return acceptedFrac / (niter * nelec);
}

template<typename Wfn, typename Walker>
void getStochasticGradientHessianDirectContinuousTime(Wfn &w, Walker& walk, double &Energy, double &stddev, VectorXd &grad, DirectLM &h, double &rk, int niter)
{
  ContinuousTime<Wfn, Walker> CTMC(w, walk, niter);
  int numVars = grad.rows();
  Energy = 0.0;
  grad.setZero();
  Eigen::VectorXd grad_ratio_bar = VectorXd::Zero(numVars);
  h.T.clear();
  h.G.clear();
  h.H.clear();
  for (int iter = 0; iter < niter; iter++)
  {
    CTMC.LocalEnergy();
    CTMC.LocalGradient(iter % (int) std::round(rk));
    CTMC.LocalEnergyGradient(iter % (int) std::round(rk));
    CTMC.MakeMove(iter % (int) std::round(rk));
    CTMC.UpdateEnergy(Energy);
    CTMC.UpdateGradient(grad, grad_ratio_bar, iter % (int) std::round(rk));
    CTMC.UpdateLM(h, iter % (int) std::round(rk));
    CTMC.UpdateBestDet();
  }
  CTMC.FinishEnergy(Energy, stddev, rk);
  CTMC.FinishGradient(grad, grad_ratio_bar, Energy);
  CTMC.FinishBestDet();
}

//############################################################Metropolis Evaluation############################################################################
template<typename Wfn, typename Walker>
double getStochasticGradientMetropolis(Wfn &w, Walker &walk, double &Energy, double &stddev, VectorXd &grad, double &rk, int niter)
{
  Metropolis<Wfn, Walker> M(w, walk, niter); 
  Energy = 0.0, stddev = 0.0, rk = 0.0;
  grad.setZero();
  VectorXd grad_ratio_bar = VectorXd::Zero(grad.rows());
  for (int iter = 0; iter < niter; iter++)
  {
    M.LocalEnergy();
    M.LocalGradient();
    M.MakeMove();
    M.UpdateEnergy(Energy);
    M.UpdateGradient(grad, grad_ratio_bar);
  }
  M.FinishEnergy(Energy, stddev, rk);
  M.FinishGradient(grad, grad_ratio_bar, Energy);
  return M.fracAmoves;
}

//############################################################Correlated Sampling Evaluation############################################################################
template<typename Wfn, typename Walker>
void CorrelatedSampling(int niter, std::vector<Eigen::VectorXd> &V, std::vector<double> &E)
{
  CorrelatedSamplingContinuousTime<Wfn, Walker> CSCT(V);
  E.assign(V.size(), 0.0);
  for (int iter = 0; iter < niter; iter++)
  {
    CSCT.LocalEnergy();
    CSCT.MakeMove();
    CSCT.UpdateEnergy(E);
  }
  CSCT.FinishEnergy(E);
}

template<typename Wfn, typename Walker>
void CorrelatedSamplingRealSpace(int niter, std::vector<Eigen::VectorXd> &V, std::vector<double> &E)
{
  int nWave = V.size();
  std::vector<Wfn> Wave;
  std::vector<Walker> Walk;
  std::vector<double> Eloc(nWave, 0.0);
  std::vector<double> T(nWave, 0.0);
  std::vector<double> cumT(nWave, 0.0);

  //init walk0
  {
    Wfn wave;
    Walker walk;
    wave.updateOptVariables(V[0]);
    wave.initWalker(walk);
    Wave.push_back(wave);
    Walk.push_back(walk);
  }
  for (int i = 1; i < nWave; i++)
  {
    Wfn wave;
    Walker walk;
    wave.updateOptVariables(V[i]);
    wave.initWalker(walk, Walk[0].d);
    Wave.push_back(wave);
    Walk.push_back(walk);
  }

  auto random = std::bind(std::uniform_real_distribution<double>(0, 1), std::ref(generator));
  int nelec = Walk[0].d.nelec;
  double effIter = 0; 
  double acceptedFrac = 0;
  for (int iter = 0; iter < niter; iter++)
  {
    //int n electron move
    for (int elec = 0; elec < nelec; elec++)
    {
      double ovlpProb, proposalProb;
      Vector3d step;
      Walk[0].getStep(step, elec, schd.rStepSize, Wave[0].getRef(), Wave[0].getCorr(), ovlpProb, proposalProb);
      step += Walk[0].d.coord[elec];
      //if move is simple or gaussian
      if (ovlpProb < -0.5) ovlpProb = std::pow(Wave[0].getOverlapFactor(elec, step, Walk[0]), 2); 
    
      //accept move for all walkers based on metropolis criteria
      if (ovlpProb * proposalProb > random())
      {
        for (int i = 0; i < nWave; i++)
        {
          Walk[i].updateWalker(elec, step, Wave[i].getRef(), Wave[i].getCorr());
        }
      }
    }

    //sample energy
    if (iter > 0.01 * niter)
    {
      effIter++;
      Eloc[0] = Wave[0].rHam(Walk[0]);
      T[0] = 1.0;
      double RefOverlap = Wave[0].Overlap(Walk[0]);
      for (int i = 1; i < nWave; i++)
      {
        Eloc[i] = Wave[i].rHam(Walk[i]);
        double Overlap = Wave[i].Overlap(Walk[i]);
        T[i] = (Overlap * Overlap) / (RefOverlap * RefOverlap);

        if (std::isnan(T[i]) || std::isnan(Eloc[i]))
        {
          if (commrank == 0 && schd.printOpt)
          {
            cout << "nan val" << endl;
            cout << Eloc[i] << endl;
            cout << Overlap * Overlap << endl;
            cout << RefOverlap * RefOverlap << endl << endl;
          }
          T[i] = 0.0;
          Eloc[i] = 0.0;
        }
      } 

      //accumulate averages
      for (int i = 0; i < E.size(); i++)
      {
        E[i] += T[i] * Eloc[i];
        cumT[i] += T[i];
      }
    }
  }

#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &(E[0]), E.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(cumT[0]), cumT.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
  std::transform(E.begin(), E.end(), cumT.begin(), E.begin(), [](double val1, double val2) -> double { return val1 /= val2; });
    //just incase any of the energies are nan
    std::transform(E.begin(), E.end(), E.begin(), [](double val) -> double { return std::isnan(val) ? 0.0 : val; });
    /*
    for (int i = 0; i < E.size(); i++)
    {
      if (commrank == 0 && std::isnan(E[i]))
      {
        cout << "nan energy value encountered during correlated sampling" << endl;
        exit(0);
      }  
    }
    */
}

template<typename Wfn, typename Walker>
class CorrSampleWrapper
{
  public:
    int sIter;
    
    CorrSampleWrapper(int _niter) : sIter(_niter) {}

    void run(std::vector<Eigen::VectorXd> &V, std::vector<double> &E)
    {
      CorrelatedSampling<Wfn, Walker>(sIter, V, E);
    };

    void runRealSpace(std::vector<Eigen::VectorXd> &V, std::vector<double> &E)
    {
      CorrelatedSamplingRealSpace<Wfn, Walker>(sIter, V, E);
    };
};

template <typename Wfn, typename Walker>
class getGradientWrapper
{
 public:
  Wfn &w;
  Walker &walk;
  int stochasticIter;
  bool ctmc;
  getGradientWrapper(Wfn &pw, Walker &pwalk, int niter, bool pctmc) : w(pw), walk(pwalk)
  {
    stochasticIter = niter;
    ctmc = pctmc;
  };

  double getGradient(VectorXd &vars, VectorXd &grad, double &E0, double &stddev, double &rt, bool deterministic)
  {
    w.updateVariables(vars);
    w.initWalker(walk);
    double acceptanceProb;
    if (!deterministic)
    {
      if (ctmc)
      {
        acceptanceProb = 1.0;
        getStochasticGradientContinuousTime(w, walk, E0, stddev, grad, rt, stochasticIter);
      }
      else
      {
        acceptanceProb = getStochasticGradientMetropolis(w, walk, E0, stddev, grad, rt, stochasticIter);
      }
    }
    else
    {
      acceptanceProb = 1.0;
      stddev = 0.0;
      rt = 1.0;
      getGradientDeterministic(w, walk, E0, grad);
    }
    w.writeWave();
    return acceptanceProb;
  };

  double getGradientRealSpace(VectorXd &vars, VectorXd &grad, double &E0, double &stddev, double &rt, bool deterministic)
  {
    double acceptedFrac;
    w.updateOptVariables(vars);
    //w.printVariables();
    //exit(0);
    w.initWalker(walk);
    if (!deterministic)
      acceptedFrac = getGradientMetropolisRealSpace(w, walk, E0, stddev, grad, rt, stochasticIter, 0.5e-3);

    w.writeWave();
    return acceptedFrac;
  };

  void getMetricRealSpace(VectorXd &vars, VectorXd &grad, VectorXd& H, DirectMetric& S, double &E0, double &stddev, double &rt, bool deterministic)
  {
    w.updateOptVariables(vars);
    w.initWalker(walk);
    if (!deterministic)
      getGradientMetricMetropolisRealSpace(w, walk, E0, stddev, grad, H, S, rt, stochasticIter, 0.5e-3);

    w.writeWave();
  };
  
  double getHessianRealSpace(VectorXd &vars, VectorXd &grad, MatrixXd& H,
                           MatrixXd& S, double &E0, double &stddev,
                           double &rt, bool deterministic)
  {
    double acceptedFrac;
    w.updateOptVariables(vars);
    w.initWalker(walk);
    if (!schd.sampleEveryRt) rt = 1.0;
    if (!deterministic)
      acceptedFrac = getGradientHessianMetropolisRealSpace(w, walk, E0, stddev, grad, H, S, rt, stochasticIter);
    
    w.writeWave();
    return acceptedFrac;
  };

  double getHessianDirectRealSpace(VectorXd &vars, VectorXd &grad, DirectLM &H, double &E0, double &stddev, double &rt, bool deterministic)
  {
    double acceptedFrac;
    w.updateOptVariables(vars);
    w.initWalker(walk);
    if (!schd.sampleEveryRt) rt = 1.0;
    if (!deterministic)
    {
      if (rt == 0.0)
        acceptedFrac = getGradientMetropolisRealSpace(w, walk, E0, stddev, grad, rt, stochasticIter, 0.5e-3);
      else
        acceptedFrac = getGradientHessianDirectMetropolisRealSpace(w, walk, E0, stddev, grad, H, rt, stochasticIter);
    }
    w.writeWave();
    return acceptedFrac;
  };
  
  void getMetric(VectorXd &vars, VectorXd &grad, VectorXd &H, DirectMetric &S, double &E0, double &stddev, double &rt, bool deterministic)
  {
      w.updateVariables(vars);
      w.initWalker(walk);
      if (!deterministic)
        getStochasticGradientMetricContinuousTime(w, walk, E0, stddev, grad, H, S, rt, stochasticIter);
      else
      {
        stddev = 0.0;
      	rt = 1.0;
      	getGradientMetricDeterministic(w, walk, E0, grad, H, S);
      }
      w.writeWave();
  };
  
/*
  void getVariance(VectorXd &vars, VectorXd &grad, DirectVarLM &H, double &Var, double &E0, double &stddev, double &rt, bool deterministic)
  {
    w.updateVariables(vars);
    w.initWalker(walk);
    if (!deterministic)
    {
      if (rt == 0.0)
        getStochasticGradientContinuousTime(w, walk, E0, stddev, grad, rt, stochasticIter);
      else 
        getStochasticGradientVarianceContinuousTime(w, walk, Var, E0, stddev, grad, H, rt, stochasticIter);
    }
    else
    {
      stddev = 0.0;
      rt = 1.0;
      cout << "Deterministic variance not yet implemented" << endl;
    }
    w.writeWave();
  };
*/

  double getHessian(VectorXd &vars, VectorXd &grad, MatrixXd &Hmatrix, MatrixXd &Smatrix, double &E0, double &stddev, double &rt, bool deterministic)
  {
    w.updateVariables(vars);
    w.initWalker(walk);
    if (!schd.sampleEveryRt) rt = 1.0;
    if (!deterministic)
    {
      getStochasticGradientHessianContinuousTime(w, walk, E0, stddev, grad, Hmatrix, Smatrix, rt, stochasticIter);
    }
    else
    {
      stddev = 0.0;
      rt = 1.0;
      getGradientHessianDeterministic(w, walk, E0, grad, Hmatrix, Smatrix);
    }
    w.writeWave();
    return 1.0; //the accepted fraction is 1
  };

  double getHessianDirect(VectorXd &vars, VectorXd &grad, DirectLM &H, double &E0, double &stddev, double &rt, bool deterministic)
  {
    w.updateVariables(vars);
    w.initWalker(walk);
    if (!schd.sampleEveryRt) rt = 1.0;
    if (!deterministic)
    {
      if (rt == 0.0)
        getStochasticGradientContinuousTime(w, walk, E0, stddev, grad, rt, stochasticIter);
      else
      {
        getStochasticGradientHessianDirectContinuousTime(w, walk, E0, stddev, grad, H, rt, stochasticIter);
      }
    }
    else
    {
      stddev = 0.0;
      rt = 1.0;
      getGradientHessianDirectDeterministic(w, walk, E0, grad, H);
    }
    w.writeWave();
    return 1.0;
  };

  double getMetricRandom(VectorXd &vars, VectorXd &grad, MatrixXd &O, MatrixXd &SO, double &E0, double &stddev, double &rt, bool deterministic)
  {
    w.updateVariables(vars);
    w.initWalker(walk);
    if (!deterministic)
    {
      if (rt == 0.0)
      {
        getStochasticGradientContinuousTime(w, walk, E0, stddev, grad, rt, stochasticIter);
      }
      else
      {
        getStochasticGradientMetricRandomContinuousTime(w, walk, E0, stddev, grad, O, SO, rt, stochasticIter);
      }
    }
    else
    {
      stddev = 0.0;
      rt = 1.0;
    }
    w.writeWave();
    return 1.0;
  };

  double getMetricRandomRealSpace(VectorXd &vars, VectorXd &grad, MatrixXd &O, MatrixXd &SO, double &E0, double &stddev, double &rt, bool deterministic)
  {
    double acceptedFrac;
    w.updateOptVariables(vars);
    w.initWalker(walk);
    if (!deterministic)
    {
      if (rt == 0.0)
      {
        acceptedFrac = getGradientMetropolisRealSpace(w, walk, E0, stddev, grad, rt, stochasticIter, 0.5e-3);
      }
      else
      {
        acceptedFrac = getGradientMetricRandomMetropolisRealSpace(w, walk, E0, stddev, grad, O, SO, rt, stochasticIter);
      }
    }
    w.writeWave();
    return acceptedFrac;
  };

  double getHessianRandom(VectorXd &vars, VectorXd &grad, MatrixXd &Q, MatrixXd &SQ, MatrixXd &HQ, double &E0, double &stddev, double &rt, bool deterministic)
  {
    w.updateVariables(vars);
    w.initWalker(walk);
    if (!schd.sampleEveryRt) rt = 1.0;
    /*
    if (!deterministic)
    {
      if (rt == 0.0)
      {
        getStochasticGradientContinuousTime(w, walk, E0, stddev, grad, rt, stochasticIter);
      }
      else
      {
        getStochasticGradientHessianRandomContinuousTime(w, walk, E0, stddev, grad, Q, HQ, SQ, rt, stochasticIter);
      }
    }
    else
    {
      rt = 1.0;
    }
    */
    getStochasticGradientHessianRandomContinuousTime(w, walk, E0, stddev, grad, Q, HQ, SQ, rt, stochasticIter);
    w.writeWave();
    return 1.0;
  };

  double getHessianRandomRealSpace(VectorXd &vars, VectorXd &grad, MatrixXd &Q, MatrixXd &SQ, MatrixXd &HQ, double &E0, double &stddev, double &rt, bool deterministic)
  {
    double acceptedFrac;
    w.updateOptVariables(vars);
    w.initWalker(walk);
    if (!schd.sampleEveryRt) rt = 1.0;
    if (!deterministic)
      acceptedFrac = getGradientHessianRandomMetropolisRealSpace(w, walk, E0, stddev, grad, Q, HQ, SQ, rt, stochasticIter);
    w.writeWave();
    return acceptedFrac;
  };

};
#endif
