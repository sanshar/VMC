#ifndef CTMC_HEADER_H
#define CTMC_HEADER_H
#include <Eigen/Dense>
#include <vector>
#include "Determinants.h"
#include "workingArray.h"
#include "statistics.h"
//#include "sr.h"
//#include "linearMethod.h"
//#include "variance.h"
#include "global.h"
//#include "evaluateE.h"
//#include "LocalEnergy.h"
#include <iostream>
#include <fstream>
#include <boost/serialization/serialization.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <algorithm>
//#include <stan/math.hpp>

#ifndef SERIAL
#include "mpi.h"
#endif

template<typename Wfn, typename Walker>
class ContinuousTime
{
  public:
  Wfn &w;
  Walker &walk;
  long numVars;
  int norbs, nalpha, nbeta;
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
    
  ContinuousTime(Wfn &_w, Walker &_walk, int niter) : w(_w), walk(_walk)
  {
    nsample = min(niter, 200000);
    numVars = w.getNumVariables();
    norbs = Determinant::norbs;
    nalpha = Determinant::nalpha;
    nbeta = Determinant::nbeta;
    bestDet = walk.getDet();
    cumT = 0.0, cumT2 = 0.0, cumT_everyrk = 0.0, S1 = 0.0, oldEnergy = 0.0, bestOvlp = 0.0; 
  }

  void LocalEnergy()
  {
    Eloc = 0.0, ovlp = 0.0;
    w.HamAndOvlp(walk, ovlp, Eloc, work);
    if (schd.debug) {
      cout << walk << endl;
      cout << "ham  " << Eloc << "  ovlp  " << ovlp << endl << endl;
    }
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
    walk.updateWalker(w.getRef(), w.getCorr(), work.excitation1[nextDet], work.excitation2[nextDet]);
  }

  void MakeMove(int sample)
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
    if (sample == 0)
        cumT_everyrk += T;
    walk.updateWalker(w.getRef(), w.getCorr(), work.excitation1[nextDet], work.excitation2[nextDet]);
  }

  void UpdateBestDet()
  {
    if (abs(ovlp) > bestOvlp)
    {
      bestOvlp = abs(ovlp);
      bestDet = walk.getDet();
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
      if (schd.printLevel > 7) cout << bestDet << endl;
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
    w.OverlapWithGradient(walk, ovlp, grad_ratio);
  }

  void LocalGradient(int sample)
  {
    if (sample == 0)
    {
      grad_ratio.setZero(numVars);
      w.OverlapWithGradient(walk, ovlp, grad_ratio);
    }
  }
  
  void UpdateGradient(Eigen::VectorXd &grad, Eigen::VectorXd &grad_ratio_bar)
  {
    grad_ratio_bar += T * (grad_ratio - grad_ratio_bar) / cumT;
    grad += T * (grad_ratio * Eloc - grad) / cumT;
  }

  void UpdateGradient(Eigen::VectorXd &grad, Eigen::VectorXd &grad_ratio_bar, int sample)
  {
    if (sample == 0)
    {
      grad_ratio_bar += T * (grad_ratio - grad_ratio_bar) / cumT_everyrk;
      grad += T * (grad_ratio * Eloc - grad) / cumT_everyrk;
    }
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

  void LocalEnergyGradient()
  {
    w.OverlapWithLocalEnergyGradient(walk, work, grad_Eloc);
  }

  void LocalEnergyGradient(int sample)
  {
    if (sample == 0)
    {
      w.OverlapWithLocalEnergyGradient(walk, work, grad_Eloc);
    }
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

/*
  void UpdateSR(double &Energy_everyrk, VectorXd &grad_everyrk, VectorXd &grad_ratio_bar_everyrk, DirectMetric &S, int rk)
  {
    if ((iter + 1) % rk == 0)
    {
      cumT_everyrk += T;
      Energy_everyrk += T * (Eloc - Energy_everyrk) / cumT_everyrk;
      grad_ratio_bar_everyrk += T * (grad_ratio - grad_ratio_bar_everyrk) / cumT_everyrk;
      grad_everyrk += T * (grad_ratio * Eloc - grad_everyrk) / cumT_everyrk;
      Eigen::VectorXd appended(numVars + 1);
      appended << 1.0, grad_ratio;
      if (schd.direct)
      {
        S.Vectors.push_back(appended);
        S.T.push_back(T);
      }
      else
      {
        S.Smatrix += T * (appended * appended.adjoint() - S.Smatrix) / cumT_everyrk;
      }
    }
  }
*/
  
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

/*
  void FinishSR(double &Energy, Eigen::VectorXd &grad, Eigen::VectorXd &grad_ratio_bar, Eigen::VectorXd &H, DirectMetric &S)
  {
#ifndef SERIAL
    MPI_Allreduce(MPI_IN_PLACE, &Energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, (grad_ratio_bar.data()), grad_ratio_bar.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, (grad.data()), grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    Energy /= commsize;
    grad /= commsize;
    grad_ratio_bar /= commsize;
#endif
    grad = (grad - Energy * grad_ratio_bar);
    H.setZero(grad.rows() + 1);
    VectorXd appended(grad.rows());
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
*/

  void UpdateLM(DirectLM &H)
  {
    Eigen::VectorXd Gappended(numVars + 1), Happended(numVars + 1), Htemp(numVars);
    Gappended << 0.0, grad_ratio;
    Htemp = grad_Eloc + Eloc * grad_ratio;
    Happended << 0.0, Htemp;
    H.H.push_back(Happended);
    H.G.push_back(Gappended);
    H.T.push_back(T);
    H.Eloc.push_back(Eloc);
  }

  void UpdateLM(DirectLM &H, int sample)
  {
    if (sample == 0)
    {
      Eigen::VectorXd Gappended(numVars + 1), Happended(numVars + 1), Htemp(numVars);
      Gappended << 0.0, grad_ratio;
      Htemp = grad_Eloc + Eloc * grad_ratio;
      Happended << 0.0, Htemp;
      H.H.push_back(Happended);
      H.G.push_back(Gappended);
      H.T.push_back(T);
      H.Eloc.push_back(Eloc);
    }
  }

  void UpdateLM(Eigen::MatrixXd &Hmatrix, Eigen::MatrixXd &Smatrix)
  {
    Eigen::VectorXd Gappended(numVars + 1), Happended(numVars + 1), Htemp(numVars);
    Gappended << 1.0, grad_ratio;
    Htemp = grad_Eloc + Eloc * grad_ratio;
    Happended << Eloc, Htemp;
    Smatrix += T * (Gappended * Gappended.adjoint() - Smatrix) / cumT_everyrk;
    Hmatrix += T * (Gappended * Happended.adjoint() - Hmatrix) / cumT_everyrk;
  }

  void UpdateLM(Eigen::MatrixXd &Hmatrix, Eigen::MatrixXd &Smatrix, int sample)
  {
    if (sample == 0)
    {
      Eigen::VectorXd Gappended(numVars + 1), Happended(numVars + 1), Htemp(numVars);
      Gappended << 1.0, grad_ratio;
      Htemp = grad_Eloc + Eloc * grad_ratio;
      Happended << Eloc, Htemp;
      Smatrix += T * (Gappended * Gappended.adjoint() - Smatrix) / cumT_everyrk;
      Hmatrix += T * (Gappended * Happended.adjoint() - Hmatrix) / cumT_everyrk;
    }
  }

  void FinishLM(Eigen::MatrixXd &Hmatrix, Eigen::MatrixXd &Smatrix)
  {
#ifndef SERIAL
      MPI_Allreduce(MPI_IN_PLACE, (Smatrix.data()), Smatrix.rows() * Smatrix.cols(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, (Hmatrix.data()), Hmatrix.rows() * Hmatrix.cols(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
      Smatrix /= commsize;
      Hmatrix /= commsize;
  }

/*
  void UpdateVariance(double &Variance, DirectVarLM &H, int rk)
  {
    Variance += T * (Eloc * Eloc - Variance) / cumT;
    if ((iter + 1) % rk == 0)
    {
      cumT_everyrk += T;
      Eigen::VectorXd Gappended(numVars + 1), Happended(numVars + 1);
      Gappended << 0.0, grad_ratio;
      Happended << 0.0, grad_Eloc;
      H.H.push_back(Happended);
      H.G.push_back(Gappended);
      H.T.push_back(T);
      H.Eloc.push_back(Eloc);

      //Variance += T * (Eloc * Eloc - Variance) / cumT_everyrk;
      Energy_everyrk += T * (Eloc - Energy_everyrk) / cumT_everyrk;
      grad_Eloc_bar += T * (grad_Eloc - grad_Eloc_bar) / cumT_everyrk;
      Eloc_grad_Eloc_bar += T * (Eloc * grad_Eloc - Eloc_grad_Eloc_bar) / cumT_everyrk;
      grad_ratio_bar += T * (grad_ratio - grad_ratio_bar) / cumT_everyrk;
      Eloc_grad_ratio_bar += T * (Eloc * grad_ratio - Eloc_grad_ratio_bar) / cumT_everyrk;
      H.grad_Eloc.push_back(grad_Eloc);
      H.T.push_back(T);
    }
  }
*/

/*
  void FinishVariance(double &Variance, double &Energy)
  {
#ifndef SERIAL
      MPI_Allreduce(MPI_IN_PLACE, &Variance, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
      Variance /= commsize;
      Variance -= Energy * Energy;
    
#ifndef SERIAL
      MPI_Allreduce(MPI_IN_PLACE, &Variance, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &Energy_everyrk, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, (grad_Eloc_bar.data()), grad_Eloc_bar.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, (Eloc_grad_Eloc_bar.data()), Eloc_grad_Eloc_bar.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, (grad_ratio_bar.data()), grad_ratio_bar.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, (Eloc_grad_ratio_bar.data()), grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      Energy_everyrk /= commsize;
      Variance /= commsize;
      grad_Eloc_bar /= commsize;
      Eloc_grad_Eloc_bar /= commsize;
      grad_ratio_bar /= commsize;
      Eloc_grad_ratio_bar /= commsize;
#endif
    Variance -= Energy_everyrk * Energy_everyrk;
    H.grad_Energy = 2.0 * (Eloc_grad_ratio_bar - Energy_everyrk * grad_ratio_bar);
    grad = 2.0 * (Eloc_grad_Eloc_bar - Energy_everyrk * grad_Eloc_bar);
  } 
*/
};

template<typename Wfn, typename Walker>
class CorrelatedSamplingContinuousTime
{
  public:
  int nWave;
  std::vector<Wfn> Wave;
  std::vector<Walker> Walk;
  std::vector<double> Eloc; 
  std::vector<double> T; //only the 0th element is a vmc weight, the rest are (Y_i)^2 / (Y_0)^2
  std::vector<double> cumT;
  workingArray RefWork, work;
  
  double random()
  {
    uniform_real_distribution<double> dist(0, 1);
    return dist(generator);
  }

  CorrelatedSamplingContinuousTime(std::vector<Eigen::VectorXd> &V)
  {
    nWave = V.size();
    Eloc.assign(nWave, 0.0);
    T.assign(nWave, 0.0);
    cumT.assign(nWave, 0.0);
    for (int i = 0; i < nWave; i++)
    {
      Wfn wave;
      Walker walk;
      wave.updateVariables(V[i]);
      wave.initWalker(walk);
      Wave.push_back(wave);
      Walk.push_back(walk);
    }
  }

  void LocalEnergy()
  {
    Eloc[0] = 0.0;
    double RefOverlap = 0.0;
    Wave[0].HamAndOvlp(Walk[0], RefOverlap, Eloc[0], RefWork);
    for (int i = 1; i < nWave; i++)
    {
      Eloc[i] = 0.0;
      double Overlap = 0.0;
      Wave[i].HamAndOvlp(Walk[i], Overlap, Eloc[i], work);
      T[i] = (Overlap * Overlap) / (RefOverlap * RefOverlap);
      if (std::isnan(T[i]) || std::isnan(Eloc[i]))
      {
          if (commrank == 0 && schd.printOpt)
          {
            cout << "nan val for wfn " << i << endl;
            cout << Eloc[i] << endl;
            cout << Overlap * Overlap << endl;
            cout << RefOverlap * RefOverlap << endl << endl;
          }
          Eloc[i] = 0.0;
          T[i] = 0.0;
      }
    }
  }

  void MakeMove()
  {
    //make move with respect to reference
    double cumOvlp = 0.0;
    for (int i = 0; i < RefWork.nExcitations; i++)
    {
      cumOvlp += abs(RefWork.ovlpRatio[i]);
      RefWork.ovlpRatio[i] = cumOvlp;
    }
    double nextDetRand = random() * cumOvlp;
    int nextDet = lower_bound(RefWork.ovlpRatio.begin(), RefWork.ovlpRatio.begin() + RefWork.nExcitations, nextDetRand) - RefWork.ovlpRatio.begin();
    //update all walkers
    T[0] = 1.0 / cumOvlp;
    Walk[0].updateWalker(Wave[0].getRef(), Wave[0].getCorr(), RefWork.excitation1[nextDet], RefWork.excitation2[nextDet]);
    for (int i = 1; i < nWave; i++)
    {  
      T[i] *= T[0];
      Walk[i].updateWalker(Wave[i].getRef(), Wave[i].getCorr(), RefWork.excitation1[nextDet], RefWork.excitation2[nextDet]);
    }
  }

  void UpdateEnergy(std::vector<double> &E)
  {
    for (int i = 0; i < E.size(); i++)
    {
      E[i] += T[i] * Eloc[i];
      cumT[i] += T[i];
    }
  } 

  void FinishEnergy(std::vector<double> &E)
  {
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

};
#endif
