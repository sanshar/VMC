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
    //cout << walk;
    w.HamAndOvlpLanczos(walk, coeffsSample, overlapSample, work, moreWork, alpha);
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
  
  int transIter = 0, nTransIter = 1000;

  while (transIter < nTransIter) {
    double cumovlpRatio = 0;
    //when using uniform probability 1./numConnection * max(1, pi/pj)
    for (int i = 0; i < work.nExcitations; i++) {
      cumovlpRatio += abs(work.ovlpRatio[i]);
      work.ovlpRatio[i] = cumovlpRatio;
    }

    double nextDetRandom = random() * cumovlpRatio;
    int nextDet = std::lower_bound(work.ovlpRatio.begin(), (work.ovlpRatio.begin() + work.nExcitations),
                                   nextDetRandom) - work.ovlpRatio.begin();

    transIter++;
    walk.updateWalker(w.getRef(), w.getCorr(), work.excitation1[nextDet], work.excitation2[nextDet]);
    w.HamAndOvlpLanczos(walk, coeffsSample, ovlpSample, work, moreWork, alpha);
  }

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
void getGradientMetropolisRealSpace(Wfn &wave, Walker &walk, double &E0, double &stddev, Eigen::VectorXd &grad, double &rk, int niter, double targetError)
{
  auto random = std::bind(std::uniform_real_distribution<double>(0, 1),
                          std::ref(generator));
  double ovlp = pow(wave.Overlap(walk), 2);
  double ham;

  rDeterminant bestDet = walk.getDet();
  double bestovlp = ovlp;

  
  Vector3d step;
  int elecToMove = 0, nelec = walk.d.nelec;

  Statistics Stats;
  
  double avgPot = 0; int iter = 0, effIter = 0, sampleSteps = nelec;
  double acceptedFrac = 0;
  double M1 = 0., S1 = 0.;
  int nstore = 1000000 / commsize;
  int corrIter = min(nstore, niter/sampleSteps);
  std::vector<double> corrError(corrIter * commsize, 0);

  VectorXd localdiagonalGrad = VectorXd::Zero(grad.rows()),
      diagonalGrad = VectorXd::Zero(grad.rows());

  vector<double> aoValues(10 * Determinant::norbs, 0.0);

  double ovlpRatio = -1.0, proposalProb;
  while (iter < niter) {
    elecToMove = iter%nelec;
    walk.getStep(step, elecToMove, schd.realSpaceStep,
                 wave.ref, wave.getCorr(), ovlpRatio, proposalProb);

    step += walk.d.coord[elecToMove];

    iter ++;
    if (iter%sampleSteps == 0 && iter > 0.01*niter) {
      ham = wave.rHam(walk);
      wave.OverlapWithGradient(walk, ovlp, localdiagonalGrad);

      for (int i = 0; i < grad.rows(); i++)
      {
        diagonalGrad[i] += (localdiagonalGrad[i] - diagonalGrad[i])/(effIter+1);
        grad[i] += (ham * localdiagonalGrad[i] - grad[i])/(effIter + 1);
        localdiagonalGrad[i] = 0.0;
      }
      double avgPotold = avgPot;
      avgPot += (ham - avgPot)/(effIter+1);
      S1 += (ham - avgPotold) * (ham - avgPot);
      if (effIter < corrIter)
        Stats.push_back(ham);
      //corrError[effIter + commrank * corrIter] = ham;
      effIter++;
    }

    

    if (ovlpRatio < -0.5)
      ovlpRatio = pow(wave.getOverlapFactor(elecToMove, step, walk), 2);
    
    if (ovlpRatio*proposalProb > random()) {
      acceptedFrac++;
      walk.updateWalker(elecToMove, step, wave.getRef(), wave.getCorr());

      ovlp = ovlp*ovlpRatio;

      if (abs(ovlp) > abs(bestovlp)) {
        bestovlp = ovlp;
        bestDet = walk.getDet();
      }
    }
    


  }
#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &(diagonalGrad[0]), grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(grad[0]), grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(corrError[0]), corrError.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &avgPot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  Stats.Block();
  rk = Stats.BlockCorrTime();
  //vector<double> b_size, r_x; vector<double> tauError(corrError.size(), 1.0);
  //block(b_size, r_x, corrError, tauError);
  //rk = corrFunc(b_size, r_x);


  double n_eff = commsize * effIter;
  S1 /= effIter;
  stddev = sqrt((S1 * rk /n_eff));
  avgPot /= commsize;
  E0 = avgPot;

  //cout << diagonalGrad<<endl<<endl;
  //cout << grad<<endl;
  //exit(0);
  diagonalGrad /= (commsize);
  grad /= (commsize);
  grad = grad - E0 * diagonalGrad;

  if (commrank == 0)
  {
    char file[5000];
    sprintf(file, "BestCoordinates.txt");
    std::ofstream ofs(file, std::ios::binary);
    boost::archive::binary_oarchive save(ofs);
    save << bestDet;
  }
  
}

template<typename Wfn, typename Walker>
void getGradientMetricMetropolisRealSpace(Wfn &wave, Walker &walk, double &E0, double &stddev, Eigen::VectorXd &grad, VectorXd& H, DirectMetric& S, double &rk, int niter, double targetError)
{
  auto random = std::bind(std::uniform_real_distribution<double>(0, 1),
                          std::ref(generator));
  double ovlp = pow(wave.Overlap(walk), 2);
  double ham;

  rDeterminant bestDet = walk.getDet();
  double bestovlp = ovlp;


  Statistics Stats;
  Vector3d step;
  int elecToMove = 0, nelec = walk.d.nelec;

  double avgPot = 0; int iter = 0, effIter = 0, sampleSteps = 2*nelec;
  
  double acceptedFrac = 0;
  double M1 = 0., S1 = 0.;
  int nstore = 1000000 / commsize;
  int corrIter = min(nstore, niter/sampleSteps);
  std::vector<double> corrError(corrIter * commsize, 0);

  int numVars = grad.rows();
  H = VectorXd::Zero(numVars + 1);
  VectorXd localdiagonalGrad = VectorXd::Zero(grad.rows()),
      diagonalGrad = VectorXd::Zero(grad.rows());

  vector<double> aoValues(10 * Determinant::norbs, 0.0);

  S.T.resize(niter/sampleSteps,0);
  S.Vectors.resize(niter/sampleSteps, VectorXd::Zero(numVars + 1));
  VectorXd appended(numVars + 1);

  double ovlpRatio = -1.0, proposalProb;
  while (iter < niter) {
    elecToMove = iter%nelec;
    walk.getStep(step, elecToMove, schd.realSpaceStep,
                 wave.ref, wave.getCorr(), ovlpRatio, proposalProb);

    step += walk.d.coord[elecToMove];

    iter ++;
    if (iter%sampleSteps == 0) {
      ham = wave.rHam(walk);
   
      wave.OverlapWithGradient(walk, ovlp, localdiagonalGrad);

      if (effIter < niter/sampleSteps) {
        appended << 1.0, localdiagonalGrad;
        S.Vectors[effIter] = appended;
        S.T[effIter] = 1.0;
      }

      
      for (int i = 0; i < grad.rows(); i++)
      {
        diagonalGrad[i] += (localdiagonalGrad[i] - diagonalGrad[i])/(effIter+1);
        grad[i] += (ham * localdiagonalGrad[i] - grad[i])/(effIter + 1);
        localdiagonalGrad[i] = 0.0;
      }
      double avgPotold = avgPot;
      avgPot += (ham - avgPot)/(effIter+1);
      S1 += (ham - avgPotold) * (ham - avgPot);
      if (effIter < corrIter)
        Stats.push_back(ham);
      //corrError[effIter + commrank * corrIter] = ham;
      effIter++;
    }

 

    if (ovlpRatio < -0.5)
      ovlpRatio = pow(wave.getOverlapFactor(elecToMove, step, walk),2);

    if (ovlpRatio*proposalProb > random()) {
      acceptedFrac++;
      walk.updateWalker(elecToMove, step, wave.getRef(), wave.getCorr());

      ovlp = ovlp*ovlpRatio;

      if (abs(ovlp) > abs(bestovlp)) {
        bestovlp = ovlp;
        bestDet = walk.getDet();
      }
    }
 


  }
#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &(diagonalGrad[0]), grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(grad[0]), grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(corrError[0]), corrError.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &avgPot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  Stats.Block();
  rk = Stats.BlockCorrTime();
  //  vector<double> b_size, r_x; vector<double> tauError(corrError.size(), 1.0);
  //block(b_size, r_x, corrError, tauError);
  //rk = corrFunc(b_size, r_x);


  double n_eff = commsize * effIter;
  S1 /= effIter;
  stddev = sqrt((S1 * rk /n_eff));
  avgPot /= commsize;
  E0 = avgPot;

  diagonalGrad /= (commsize);
  grad /= (commsize);
  grad = grad - E0 * diagonalGrad;

  
  //VectorXd appended(numVars);
  //appended = diagonalGrad - schd.stepsize * grad;
  H << 1.0, (diagonalGrad - schd.stepsize * grad);

  if (commrank == 0)
  {
    char file[5000];
    sprintf(file, "BestCoordinates.txt");
    std::ofstream ofs(file, std::ios::binary);
    boost::archive::binary_oarchive save(ofs);
    save << bestDet;
  }

}



template<typename Wfn, typename Walker>
double getGradientHessianMetropolisRealSpace(Wfn &wave, Walker &walk, double &E0, double &stddev,
                                           Eigen::VectorXd &grad, MatrixXd& Hessian,
                                           MatrixXd& Smatrix, double &rk,
                                           int niter, double targetError)
{
  auto random = std::bind(std::uniform_real_distribution<double>(0, 1),
                          std::ref(generator));

  double ovlp =  1.0;//wave.Overlap(walk);;
  double ham;

  rDeterminant bestDet = walk.getDet();
  double bestovlp = ovlp; //cout << bestovlp <<endl;


  Statistics Stats;
  Vector3d step;
  int elecToMove = 0, nelec = walk.d.nelec;

  double avgPot = 0; int iter = 0, effIter = 0, sampleSteps = nelec;
  
  double acceptedFrac = 0;
  double M1 = 0., S1 = 0.;
  int nstore = 1000000 / commsize;
  int corrIter = min(nstore, niter/sampleSteps);
  std::vector<double> corrError(corrIter * commsize, 0);

  int numVars = grad.rows();
  Hessian = MatrixXd::Zero(numVars + 1, numVars+1);
  Smatrix = MatrixXd::Zero(numVars + 1, numVars+1);

  VectorXd localdiagonalGrad = VectorXd::Zero(grad.rows()+1),
      hamRatio = VectorXd::Zero(grad.rows()+1),
      diagonalGrad = VectorXd::Zero(grad.rows());

  double ovlpRatio = -1.0, proposalProb;
  while (iter < niter) {
    elecToMove = iter%nelec;

    walk.getStep(step, elecToMove, schd.realSpaceStep,
                 wave.ref, wave.getCorr(), ovlpRatio, proposalProb);

    step += walk.d.coord[elecToMove];

    iter ++;
    if (iter%sampleSteps == 0 && iter > 0.01*niter) {
      ham = wave.HamOverlap(walk, localdiagonalGrad, hamRatio);

      Hessian.noalias() += (localdiagonalGrad * hamRatio.transpose()-Hessian)/(effIter+1);
      Smatrix.noalias() += (localdiagonalGrad * localdiagonalGrad.transpose()-Smatrix)/(effIter+1);

      hamRatio[0] = 0.0; localdiagonalGrad[0] = 0.0;
      for (int i = 0; i < grad.rows(); i++)
      {
        diagonalGrad[i] += (localdiagonalGrad[i+1] - diagonalGrad[i])/(effIter+1);
        grad[i] += (ham * localdiagonalGrad[i+1] - grad[i])/(effIter + 1);
        localdiagonalGrad[i+1] = 0.0;
        hamRatio[i+1] = 0;
      }
      double avgPotold = avgPot;
      avgPot += (ham - avgPot)/(effIter+1);
      S1 += (ham - avgPotold) * (ham - avgPot);
      if (effIter < corrIter)
        Stats.push_back(ham);

      effIter++;
    }

 
    if (ovlpRatio < -0.5) 
      ovlpRatio = pow(wave.getOverlapFactor(elecToMove, step, walk), 2);
    
    if (ovlpRatio*proposalProb > random()) {
      acceptedFrac++;
      walk.updateWalker(elecToMove, step, wave.getRef(), wave.getCorr());

      ovlp = ovlp*sqrt(ovlpRatio);

      if (abs(ovlp) > abs(bestovlp)) {
        bestovlp = ovlp;
        bestDet = walk.getDet();
      }
    }
 


  }
#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &(diagonalGrad[0]), grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(grad[0]), grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(corrError[0]), corrError.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &avgPot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(Hessian(0,0)), Hessian.rows()*Hessian.cols(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(Smatrix(0,0)), Smatrix.rows()*Smatrix.cols(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  Stats.Block();
  rk = Stats.BlockCorrTime();


  double n_eff = commsize * effIter;
  S1 /= effIter;
  stddev = sqrt((S1 * rk /n_eff));
  avgPot /= commsize;
  E0 = avgPot;

  diagonalGrad /= (commsize);
  grad /= (commsize);
  grad = grad - E0 * diagonalGrad;
  Hessian = Hessian/(commsize);
  Smatrix = Smatrix/(commsize);


  if (commrank == 0)
  {
    char file[5000];
    sprintf(file, "BestCoordinates.txt");
    std::ofstream ofs(file, std::ios::binary);
    boost::archive::binary_oarchive save(ofs);
    save << bestDet;
  }

  return acceptedFrac/niter;
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
    CTMC.LocalGradient();
    CTMC.LocalEnergyGradient();
    CTMC.MakeMove();
    CTMC.UpdateEnergy(Energy);
    CTMC.UpdateGradient(grad, grad_ratio_bar);
    CTMC.UpdateLM(Hmatrix, Smatrix);
    CTMC.UpdateBestDet();
  }
  CTMC.FinishEnergy(Energy, stddev, rk);
  CTMC.FinishGradient(grad, grad_ratio_bar, Energy);
  CTMC.FinishLM(Hmatrix, Smatrix);
  CTMC.FinishBestDet();
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
    CTMC.LocalGradient();
    CTMC.LocalEnergyGradient(std::round(rk));
    CTMC.MakeMove();
    CTMC.UpdateEnergy(Energy);
    CTMC.UpdateGradient(grad, grad_ratio_bar);
    CTMC.UpdateLM(h, std::round(rk));
    CTMC.UpdateBestDet();
  }
  CTMC.FinishEnergy(Energy, stddev, rk);
  CTMC.FinishGradient(grad, grad_ratio_bar, Energy);
  CTMC.FinishBestDet();
}

//############################################################Metropolis Evaluation############################################################################
template<typename Wfn, typename Walker>
void getStochasticGradientMetropolis(Wfn &w, Walker &walk, double &Energy, double &stddev, VectorXd &grad, double &rk, int niter)
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
  for (int i = 0; i < nWave; i++)
  {
    Wfn wave;
    Walker walk;
    wave.updateVariables(V[i]);
    wave.initWalker(walk);
    Wave.push_back(wave);
    Walk.push_back(walk);
  }
  auto random = std::bind(std::uniform_real_distribution<double>(0, 1), std::ref(generator));
  Vector3d step;
  int nelec = Walk[0].d.nelec;
  double effIter = 0; 
  double acceptedFrac = 0;
  for (int iter = 0; iter < niter; iter++)
  {
    int elecToMove = iter % nelec;
    double ovlpRatio, proposalProb;
    Walk[0].getStep(step, elecToMove, schd.realSpaceStep, Wave[0].getRef(), Wave[0].getCorr(), ovlpRatio, proposalProb);
    if (ovlpRatio < -0.5) 
      ovlpRatio = pow(Wave[0].getOverlapFactor(elecToMove, step, Walk[0]), 2); 
    step += Walk[0].d.coord[elecToMove];
    if ((iter + 1) % nelec == 0 && (iter + 1) > 0.01 * niter)
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
      for (int i = 0; i < E.size(); i++)
      {
        E[i] += T[i] * Eloc[i];
        cumT[i] += T[i];
      }
    }
    if (ovlpRatio * proposalProb > random())
    {
      for (int i = 0; i < nWave; i++)
      {
        Walk[i].updateWalker(elecToMove, step, Wave[i].getRef(), Wave[i].getCorr());
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

  void getGradient(VectorXd &vars, VectorXd &grad, double &E0, double &stddev, double &rt, bool deterministic)
  {
    w.updateVariables(vars);
    w.initWalker(walk);
    if (!deterministic)
    {
      if (ctmc)
      {
        getStochasticGradientContinuousTime(w, walk, E0, stddev, grad, rt, stochasticIter);
      }
      else
      {
        getStochasticGradientMetropolis(w, walk, E0, stddev, grad, rt, stochasticIter);
      }
    }
    else
    {
      stddev = 0.0;
      rt = 1.0;
      getGradientDeterministic(w, walk, E0, grad);
    }
    w.writeWave();
  };

  void getGradientRealSpace(VectorXd &vars, VectorXd &grad, double &E0, double &stddev, double &rt, bool deterministic)
  {
    w.updateVariables(vars);
    //w.printVariables();
    //exit(0);
    w.initWalker(walk);
    if (!deterministic)
      getGradientMetropolisRealSpace(w, walk, E0, stddev, grad, rt, stochasticIter, 0.5e-3);

    w.writeWave();
  };

  void getMetricRealSpace(VectorXd &vars, VectorXd &grad, VectorXd& H, DirectMetric& S, double &E0, double &stddev, double &rt, bool deterministic)
  {
    w.updateVariables(vars);
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
    w.updateVariables(vars);
    w.initWalker(walk);
    if (!deterministic)
      acceptedFrac = getGradientHessianMetropolisRealSpace(w, walk, E0, stddev, grad,
                                            H, S, rt, stochasticIter, 0.5e-3);
    
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
};
#endif
