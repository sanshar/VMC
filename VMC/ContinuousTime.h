#ifndef CTMC_HEADER_H
#define CTMC_HEADER_H
#include <Eigen/Dense>
#include <vector>
#include "Determinants.h"
#include "workingArray.h"
#include "statistics.h"
#include "sr.h"
#include "global.h"
#include "evaluateE.h"
#include "LocalEnergy.h"
#include <iostream>
#include <fstream>
#include <boost/serialization/serialization.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <algorithm>
#include <stan/math.hpp>

#ifndef SERIAL
#include "mpi.h"
#endif

template<typename Wfn, typename Walker>
class ContinuousTime
{
  public:
  Wfn *w;
  Walker *walk;
  long numVars;
  int iter, norbs, nalpha, nbeta;
  workingArray work;
  double T, Eloc, ovlp;
  double S1, oldEnergy;
  double cumT, cumT2, cumT_everyrk;
  Eigen::VectorXd grad_ratio, grad_Eloc;
  int nsample;
  Statistics Stats; //this is only used to calculate autocorrelation length
  Determinant bestDet;
  double bestOvlp;
  
  double random()
  {
    uniform_real_distribution<double> dist(0,1);
    return dist(generator);
  }
    
  ContinuousTime(Wfn &_w, Walker &_walk, int niter) : w(&_w), walk(&_walk)
  {
    nsample = min(niter, 200000);
    numVars = w->getNumVariables();
    norbs = Determinant::norbs;
    nalpha = Determinant::nalpha;
    nbeta = Determinant::nbeta;
    bestDet = walk->getDet();
    cumT = 0.0, cumT2 = 0.0, cumT_everyrk = 0.0, S1 = 0.0, oldEnergy = 0.0, bestOvlp = 0.0; 
    iter = 0;
  }

  void LocalEnergy()
  {
    Eloc = 0.0, ovlp = 0.0;
    w->HamAndOvlp(*walk, ovlp, Eloc, work);
    iter++;
  }

  void MakeMove()
  {
    double cumOvlp = 0.0;
    for (int i = 0; i < work.nExcitations; i++)
    {
      cumOvlp += abs(work.ovlpRatio[i]);
      work.ovlpRatio[i] = cumOvlp;
    }
    T = 1.0 / cumOvlp;
    double nextDetRand = random() * cumOvlp;
    int nextDet = lower_bound(work.ovlpRatio.begin(), work.ovlpRatio.begin() + work.nExcitations, nextDetRand) - work.ovlpRatio.begin();
    cumT += T;
    cumT2 += T * T;
    walk->updateWalker(w->getRef(), w->getCorr(), work.excitation1[nextDet], work.excitation2[nextDet]);
  }

  void UpdateBestDet()
  {
    if (abs(ovlp) > bestOvlp)
    {
      bestOvlp = abs(ovlp);
      bestDet = walk->getDet();
    }
  }
  
  void FinishBestDet()
  {
    if (commrank == 0)
    {
      char file[50];
      sprintf(file, "BestDeterminant.txt");
      std::ofstream ofs(file, std::ios::binary);
      boost::archive::binary_oarchive save(ofs);
      save << bestDet;
    }
  }

  void UpdateEnergy(double &Energy)
  {
    oldEnergy = Energy;
    Energy += T * (Eloc - Energy) / cumT;
    S1 += T * (Eloc - oldEnergy) * (Eloc - Energy);
    if (Stats.X.size() < nsample)
    {
      Stats.push_back(Eloc, T);
    }
  }
  
  void FinishEnergy(double &Energy, double &stddev, double &rk)
  {
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
/*
    if (commrank == 0)
    {
      Stats.WriteBlock();
      cout << "Block rk:\t" << rk << endl;
    }
    Stats.CorrFunc();
    rk = Stats.IntCorrTime();
    if (commrank == 0)
    {
      Stats.WriteCorrFunc();
      cout << "CorrFunc rk:\t" << rk << endl;
    }
    rk = calcTcorr(Stats.X);
    if (commrank == 0)
    {
      cout << "OldCorrFunc rk:\t" << rk << endl;
    }
*/
    S1 /= cumT;
#ifndef SERIAL
    MPI_Allreduce(MPI_IN_PLACE, &Energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &S1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &rk, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &cumT, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &cumT2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    Energy /= commsize;
    S1 /= commsize;
    rk /= commsize;
    cumT /= commsize;
    cumT2 /= commsize;
#endif
    double neff = commsize * (cumT * cumT) / cumT2;
    stddev = sqrt(rk * S1 / neff);
  }

  void LocalGradient()
  {
    grad_ratio.setZero(numVars);
    w->OverlapWithGradient(*walk, ovlp, grad_ratio);
  }
  
  void UpdateGradient(Eigen::VectorXd &grad, Eigen::VectorXd &grad_ratio_bar)
  {
    grad_ratio_bar += T * (grad_ratio - grad_ratio_bar) / cumT;
    grad += T * (grad_ratio * Eloc - grad) / cumT;
  }

  void FinishGradient(Eigen::VectorXd &grad, Eigen::VectorXd &grad_ratio_bar, const double &Energy)
  {
#ifndef SERIAL
    MPI_Allreduce(MPI_IN_PLACE, (grad_ratio_bar.data()), grad_ratio_bar.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, (grad.data()), grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    grad /= commsize;
    grad_ratio_bar /= commsize;
#endif
    grad = (grad - Energy * grad_ratio_bar);
  }

  void LocalEnergyGradient(int rk)
  {
    if ((iter + 1) % rk == 0)
    {
      LocalEnergySolver Solver(walk->d);
      Eigen::VectorXd vars;
      w->getVariables(vars);
      //cout << Eloc;
      Eloc = 0.0;
      stan::math::gradient(Solver, vars, Eloc, grad_Eloc);
    }
    //cout << "\t|\t" << Eloc << endl;

    //below is very expensive and used only for debugging
/*
    Eigen::VectorXd finiteGradEloc = Eigen::VectorXd::Zero(vars.size());
    for (int i = 0; i < vars.size(); ++i)
    {
      double dt = 0.00001;
      Eigen::VectorXd varsdt = vars;
      varsdt(i) += dt;
      finiteGradEloc(i) = (Solver(varsdt) - ElocTest) / dt;
    }
    for (int i = 0; i < vars.size(); ++i)
    {
      cout << finiteGradEloc(i) << "\t" << grad_Eloc(i) << endl;
    }
*/
  }

  void UpdateSR(DirectMetric &S)
  {
    Eigen::VectorXd appended(numVars + 1);
    appended << 1.0, grad_ratio;
    if (schd.direct)
    {
      S.Vectors.push_back(appended);
      S.T.push_back(T);
    }
    else
    {
      S.Smatrix += T * (appended * appended.adjoint() - S.Smatrix) / cumT;
    }
  }
  
  void FinishSR(const Eigen::VectorXd &grad, const Eigen::VectorXd &grad_ratio_bar, Eigen::VectorXd &H, DirectMetric &S)
  {
    H.setZero(grad.rows() + 1);
    Eigen::VectorXd appended(grad.rows());
    appended = grad_ratio_bar - schd.stepsize * grad;
    H << 1.0, appended;
    if (!(schd.direct))
    {
#ifndef SERIAL
      MPI_Allreduce(MPI_IN_PLACE, (S.Smatrix.data()), S.Smatrix.rows() * S.Smatrix.cols(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
      S.Smatrix /= commsize;
    }
  }

  void UpdateVariance(double &Variance, Eigen::VectorXd &grad_Eloc_bar, Eigen::VectorXd &Eloc_grad_Eloc_bar, int rk)
  {
    if ((iter + 1) % rk == 0)
    {
      cumT_everyrk += T;
      Variance += T * (Eloc * Eloc - Variance) / cumT_everyrk;
      grad_Eloc_bar += T * (grad_Eloc - grad_Eloc_bar) / cumT_everyrk;
      Eloc_grad_Eloc_bar += T * (Eloc * grad_Eloc - Eloc_grad_Eloc_bar) / cumT_everyrk;
    }
  }

  void FinishVariance(const double &Energy, double &Variance, Eigen::VectorXd &grad, const Eigen::VectorXd &grad_Eloc_bar, const Eigen::VectorXd &Eloc_grad_Eloc_bar)
  {
    Variance = Variance - Energy * Energy;
    grad = 2.0 * (Eloc_grad_Eloc_bar - Energy * grad_Eloc_bar);
    //grad = 2.0 * Eloc_grad_Eloc_bar;
#ifndef SERIAL
      MPI_Allreduce(MPI_IN_PLACE, (grad.data()), grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &Variance, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      grad /= commsize;
      Variance /= commsize;
#endif
  }
};
#endif
