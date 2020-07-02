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
#ifndef relEvalE_HEADER_H
#define relEvalE_HEADER_H
#include <Eigen/Dense>
#include <vector>
#include "Determinants.h"
#include "relDeterminants.h"
#include "rDeterminants.h"
#include "workingArray.h"
#include "statistics.h"
#include "sr.h"
#include "linearMethod.h"
#include "global.h"
#include "relDeterministic.h"
#include "relContinuousTime.h"
#include "relContinuousTimeExcitedStates.h"
#include "Metropolis.h"
#include <iostream>
#include <fstream>
#include "iowrapper.h"
#include <boost/serialization/serialization.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <algorithm>
#include "boost/format.hpp"
#include <boost/algorithm/string.hpp>
#include <math.h>

#ifndef SERIAL
#include "mpi.h"
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif


using namespace Eigen;
using namespace std;
using namespace boost;

class oneInt;
class oneIntSOC;
class twoInt;
class twoIntHeatBathSHM;



//############################################################Deterministic Evaluation############################################################################

template<typename Wfn, typename Walker>
void relGetEnergyDeterministic(Wfn &w, Walker& walk, std::complex<double> &Energy)
{
  relDeterministic<Wfn, Walker> D(w, walk);
  Energy = 0.0 + 0.0i;
  if (commrank == 0) cout << "alldets num " << D.allDets.size() << " commsize " << commsize << endl; 
  for (int i = commrank; i < D.allDets.size(); i += commsize)
  {
    D.LocalEnergy(D.allDets[i]);
    D.UpdateEnergy(Energy);
  }
  D.FinishEnergy(Energy);
}





template<typename Wfn, typename Walker>
void relGetGradientDeterministic(Wfn &w, Walker &walk, std::complex<double> &Energy, VectorXd &grad)
{
  relDeterministic<Wfn, Walker> D(w, walk);
  Energy = 0.0 + 0.0i;
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
double relGetGradientPenaltyDeterministic(Wfn &w, Walker &walk, std::complex<double> &Energy, VectorXd &grad, std::vector<Wfn> &ls)
{
  double OP = 0;
  std::vector<std::complex<double>> ovlpPen;
  std::vector<VectorXd> gradPen;
  for (int i=0; i<schd.excitedState; i++){
     ovlpPen.push_back(0.0);
     gradPen.push_back(VectorXd::Zero(grad.rows()));
  }
  Energy = 0.0 + 0.0i;
  relDeterministic<Wfn, Walker> D(w, walk);
  grad.setZero();
  VectorXd grad_ratio_bar = VectorXd::Zero(grad.rows());
  for (int i = commrank; i < D.allDets.size(); i += commsize)
  {
    // Overlap Penalty
    D.LocalOverlap(D.allDets[i], ls);
    // Local Energy
    D.LocalEnergy(D.allDets[i]);
    D.LocalGradient();
    D.UpdateEnergy(Energy);
    D.UpdateGradient(grad, grad_ratio_bar);
    D.UpdateOverlap(ovlpPen);
    D.UpdateGradientPenalty(gradPen, ovlpPen);
  }
  D.FinishEnergy(Energy); 
  D.FinishOverlap(ovlpPen); 
  D.FinishGradient(grad, grad_ratio_bar, Energy);
  D.FinishGradientPenalty(gradPen, grad_ratio_bar, ovlpPen);
  D.CombineGradients(grad, gradPen);
  if (0==0) {
    cout << "OvlpPen with the " << schd.excitedState << " lower lying states: ";
    for (int i=0; i<schd.excitedState; i++){
      cout << std::abs(ovlpPen[i]) << "   " ;
    }
    cout << endl;
  } 
  return std::abs(ovlpPen[0]);
}


//############################################################Continuous Time Evaluation############################################################################
template<typename Wfn, typename Walker> 
void relGetStochasticEnergyContinuousTime(Wfn &w, Walker &walk, std::complex<double> &Energy, double &stddev, double &rk, int niter)
{
  relContinuousTime<Wfn, Walker> CTMC(w, walk, niter);
  Energy = 0.0 + 0.0i, stddev = 0.0, rk = 0.0;
  CTMC.LocalEnergy();
  for (int iter = 0; iter < niter; iter++)
  {
    CTMC.MakeMove();
    CTMC.UpdateEnergy(Energy);
    CTMC.LocalEnergy();
  }
  CTMC.FinishEnergy(Energy, stddev, rk);
};

template<typename Wfn, typename Walker> 
void relCompEnergyDeterministicContinuousTime(Wfn &w, Walker &walk, std::complex<double> &Energy, double &stddev, double &rk, int niter)
{
  relContinuousTime<Wfn, Walker> CTMC(w, walk, niter);
  relDeterministic<Wfn, Walker> D(w, walk);
  Energy = 0.0 + 0.0i, stddev = 0.0, rk = 0.0;
  CTMC.LocalEnergy();
  D.LocalEnergy(walk.d);
  for (int iter = 0; iter < niter; iter++)
  {
    CTMC.MakeMove();
    CTMC.UpdateEnergy(Energy);
    CTMC.LocalEnergy();
    //D.LocalEnergy(walk.d);
    //if (abs(D.Eloc-CTMC.Eloc)>1.0e-10){
    if (iter==100){
      cout << "First real diff " << walk.d << endl;
      schd.ifSOC = true;
      D.LocalEnergy(walk.d);
      CTMC.LocalEnergy();
    }
  }
  CTMC.FinishEnergy(Energy, stddev, rk);
};


template<typename Wfn, typename Walker>
void relGetStochasticEnergyContinuousTimeRandIni(Wfn &w, Walker &walk, std::complex<double> &Energy, double &stddev, double &rk, int niter)
{
  auto random = std::bind(std::uniform_real_distribution<double>(0, 1), std::ref(generator));
  relDeterministic<Wfn, Walker> D(w, walk);
  int rand = int(random()*D.allDets.size()); 
  relDeterminant initial = D.allDets[rand];
  relContinuousTime<Wfn, Walker> CTMC(w, walk, niter);
  walk.d = initial;
  Energy = 0.0 + 0.0i, stddev = 0.0, rk = 0.0;
  CTMC.LocalEnergy();
  for (int iter = 0; iter < niter; iter++)
  {
    CTMC.MakeMove();
    CTMC.UpdateEnergy(Energy);
    CTMC.LocalEnergy();
  }
  CTMC.FinishEnergy(Energy, stddev, rk);
};


    
template<typename Wfn, typename Walker>
void relGetStochasticGradientContinuousTime(Wfn &w, Walker &walk, std::complex<double> &Energy, double &stddev, VectorXd &grad, double &rk, int niter)
{
  relContinuousTime<Wfn, Walker> CTMC(w, walk, niter);
  Energy = 0.0 + 0.0i, stddev = 0.0, rk = 0.0;
  grad.setZero();
  VectorXd grad_ratio_bar = VectorXd::Zero(grad.rows());
  CTMC.LocalEnergy();
  CTMC.LocalGradient();
  for (int iter = 0; iter < niter; iter++)  // EDIT DO: during the loop, the energy should be complex
  {
    CTMC.MakeMove();
    CTMC.UpdateEnergy(Energy);
    CTMC.UpdateGradient(grad, grad_ratio_bar);
    CTMC.LocalEnergy();
    CTMC.LocalGradient();
    CTMC.UpdateBestDet();
  }
  CTMC.FinishEnergy(Energy, stddev, rk); // EDIT DO: now only the real part should be taken into account
  CTMC.FinishGradient(grad, grad_ratio_bar, Energy.real()); // EDIT DO: now only the real part should be taken into account
  CTMC.FinishBestDet();
}



template<typename Wfn, typename Walker>
double relGetStochasticGradientPenaltyContinuousTime(Wfn &w, Walker &walk, std::complex<double> &Energy, double &stddev, VectorXd &grad, double &rk, int niter, std::vector<Wfn> &ls)
{
  double OP = 0;
  std::vector<std::complex<double>> ovlpPen;
  std::vector<VectorXd> gradPen;
  for (int i=0; i<schd.excitedState; i++){
     ovlpPen.push_back(0.0);
     gradPen.push_back(VectorXd::Zero(grad.rows()));
  }
  relContinuousTimeExcitedStates<Wfn, Walker> CTMC(w, walk, niter);
  Energy = 0.0 + 0.0i, stddev = 0.0, rk = 0.0;
  grad.setZero();
  VectorXd grad_ratio_bar = VectorXd::Zero(grad.rows());
  CTMC.LocalOverlap(ls);
  CTMC.LocalEnergy();
  CTMC.LocalGradient();
  for (int iter = 0; iter < niter; iter++)  // EDIT DO: during the loop, the energy should be complex
  {
    CTMC.MakeMove();
    CTMC.UpdateEnergy(Energy);
    CTMC.UpdateGradient(grad, grad_ratio_bar);
    CTMC.UpdateOverlap(ovlpPen);
    CTMC.UpdateGradientPenalty(gradPen, ovlpPen);
    CTMC.LocalOverlap(ls);
    CTMC.LocalEnergy();
    CTMC.LocalGradient();
    CTMC.UpdateBestDet();
  }
  CTMC.FinishEnergy(Energy, stddev, rk); // EDIT DO: now only the real part should be taken into account
  CTMC.FinishGradient(grad, grad_ratio_bar, Energy.real()); // EDIT DO: now only the real part should be taken into account
  CTMC.FinishOverlap(ovlpPen); 
  CTMC.FinishGradientPenalty(gradPen, grad_ratio_bar, ovlpPen);
  CTMC.FinishBestDet();
  CTMC.CombineGradients(grad, gradPen);
  if (0==0) {
    cout << "OvlpPen with the " << schd.excitedState << " lower lying states: ";
    for (int i=0; i<schd.excitedState; i++){
      cout << std::abs(ovlpPen[i]) << "   " ;
    }
    cout << endl;
  } 
  return std::abs(ovlpPen[0]);
}


template <typename Wfn, typename Walker>
class relGetGradientWrapper
{
 public:
  Wfn &w;
  Walker &walk;
  int stochasticIter;
  bool ctmc;
  relGetGradientWrapper(Wfn &pw, Walker &pwalk, int niter, bool pctmc) : w(pw), walk(pwalk)
  {
    stochasticIter = niter;
    ctmc = pctmc;
  };

  double getGradient(VectorXd &vars, VectorXd &grad, std::complex<double> &E0, double &stddev, double &rt, bool deterministic)
  {
    w.updateVariables(vars);
    w.initWalker(walk);
    if (!deterministic)
    {
      if (ctmc)
      {
        relGetStochasticGradientContinuousTime(w, walk, E0, stddev, grad, rt, stochasticIter);
      }
      else
      {
        //getStochasticGradientMetropolis(w, walk, E0, stddev, grad, rt, stochasticIter);
        cout << "not implemented yet 0" << endl;
        exit (0);
      }
    }
    else
    {
      //cout << "not implemented yet 1" << endl;
      //exit (0);
      stddev = 0.0;
      rt = 1.0;
      relGetGradientDeterministic(w, walk, E0, grad);
    }
    w.writeWave();
    return 1.0;
  };


  double getGradientPenalty(VectorXd &vars, VectorXd &grad, std::complex<double> &E0, double &stddev, double &rt, bool deterministic)
  {
    w.updateVariables(vars);
    w.initWalker(walk);
    if (!deterministic)
    {
      if (ctmc)
      {
        cout << "not implemented yet 0" << endl;
        exit (0);
        //relGetStochasticGradientContinuousTime(w, walk, E0, stddev, grad, rt, stochasticIter);
      }
      else
      {
        //getStochasticGradientMetropolis(w, walk, E0, stddev, grad, rt, stochasticIter);
        cout << "not implemented yet 0" << endl;
        exit (0);
      }
    }
    else
    {
      //cout << "not implemented yet 1" << endl;
      //exit (0);
      stddev = 0.0;
      rt = 1.0;
      relGetGradientPenaltyDeterministic(w, walk, E0, grad);
    }
    w.writeWave();
    return 1.0;
  };



  double getGradientRealSpace(VectorXd &vars, VectorXd &grad, double &E0, double &stddev, double &rt, bool deterministic)
  {
    double acceptedFrac;
    w.updateVariables(vars);
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
      acceptedFrac = getGradientHessianMetropolisRealSpace(w, walk, E0, stddev, grad, H, S, rt, stochasticIter);
    
    w.writeWave();
    return acceptedFrac;
  };

  double getHessianDirectRealSpace(VectorXd &vars, VectorXd &grad, DirectLM &H, double &E0, double &stddev, double &rt, bool deterministic)
  {
    double acceptedFrac;
    w.updateVariables(vars);
    w.initWalker(walk);
    rt = 1.0;
    if (!deterministic)
      acceptedFrac = getGradientHessianDirectMetropolisRealSpace(w, walk, E0, stddev, grad, H, rt, stochasticIter);
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



template <typename Wfn, typename Walker>
class relGetEnergyWrapper
{
  public:
    Wfn &w;
    Walker &walk;
    int stochasticIter = 0;
    bool deterministic;

    relGetEnergyWrapper(Wfn &pw, Walker &pwalk, bool pdet) : w(pw), walk(pwalk)
    {
      deterministic = pdet;
      if (pdet == false) stochasticIter = schd.stochasticIter;
    };

    double getEnergy(VectorXd &vars, std::complex<double> &E0, double &stddev, double &rt)
    {
      w.updateVariables(vars);
      w.initWalker(walk);
      if (deterministic)
      {
        relGetEnergyDeterministic(w, walk, E0);
        stddev=0.0;
        rt=1.0;
      }
      else
      {
        relGetStochasticEnergyContinuousTime(w, walk, E0, stddev, rt, stochasticIter);
      }
      return 1.0;
  };
};

class eneOnly{
  public:
    int stochasticIter;
    bool deterministic;

    eneOnly (bool det) { deterministic = det; };

    template<typename Function>
    void relGetEnergy (VectorXd& vars, Function& getEne) {
      double acceptedFrac;
      std::complex<double> E0 =0.0 + 0.0i;
      double stddev = 0.0, rt = 1.0;
      
      acceptedFrac = getEne (vars, E0, stddev, rt);

      if (commrank==0 && schd.deterministic==false) cout << "Stochastic energy: " << E0 << " (" << stddev << ")" << endl;
      else if (commrank==0) cout << "Deterministic energy: " << E0 << endl;
    };
};



template <typename Wfn, typename Walker>
class relGetGradientPenaltyWrapper
{
 public:
  Wfn &w;
  Walker &walk;
  std::vector<Wfn> &ls;
  int stochasticIter;
  bool ctmc;
  relGetGradientPenaltyWrapper(Wfn &pw, Walker &pwalk, int niter, bool pctmc, std::vector<Wfn> &pls) : w(pw), walk(pwalk), ls(pls)
  {
    stochasticIter = niter;
    ctmc = pctmc;
  };

  double getGradientPenalty(VectorXd &vars, VectorXd &grad, std::complex<double> &Energy, double &stddev, double &rt, bool deterministic)
  {
    w.updateVariables(vars);
    w.initWalker(walk);
    double OP = 0;
    if (!deterministic)
    {
      if (ctmc)
      {
        relGetStochasticGradientPenaltyContinuousTime(w, walk, Energy, stddev, grad, rt, stochasticIter, ls);
      }
      else
      {
        cout << "not implemented yet 0" << endl;
        exit (0);
      }
    }
    else
    {
      stddev = 0.0;
      rt = 1.0;
      OP = relGetGradientPenaltyDeterministic(w, walk, Energy, grad, ls);
    }
    w.writeWave();
    return OP;
  };

};




#endif
