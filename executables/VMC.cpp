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
#include <algorithm>
#include <random>
#include <chrono>
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#ifndef SERIAL
//#include "mpi.h"
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif
#include <boost/function.hpp>
#include <boost/functional.hpp>
#include <boost/bind.hpp>
#include "evaluateE.h"
#include "Determinants.h"
#include "CPSSlater.h"
#include "HFWalker.h"
#include "AGP.h"
//#include "AGPWalker.h"
#include "input.h"
#include "integral.h"
#include "SHCIshm.h"
#include "math.h"
#include "Profile.h"
#include "CIWavefunction.h"
#include "runVMC.h"
#include "Lanczos.h"

using namespace Eigen;
using namespace boost;
using namespace std;


int main(int argc, char *argv[])
{

#ifndef SERIAL
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;
#endif
  startofCalc = getTime();

  initSHM();
  //license();

  string inputFile = "input.dat";
  if (argc > 1)
    inputFile = string(argv[1]);
  readInput(inputFile, schd, false);

  generator = std::mt19937(schd.seed + commrank);

  readIntegralsAndInitializeDeterminantStaticVariables("FCIDUMP");


  //calculate the hessian/gradient
  if (schd.wavefunctionType == "CPSSlater") {
    CPSSlater<CPS, Slater> wave; HFWalker<CPS, Slater> walk;
    //CPSSlater<Jastrow, Slater> wave; HFWalker<Jastrow, Slater> walk;
    runVMC(wave, walk);
  }

  else if (schd.wavefunctionType == "JastrowSlater") {
    CPSSlater<Jastrow, Slater> wave; HFWalker<Jastrow, Slater> walk;
    runVMC(wave, walk);
  }
  

  else if (schd.wavefunctionType == "CICPSSlater") {
    CIWavefunction<CPSSlater<CPS, Slater>, HFWalker<CPS, Slater>, SpinFreeOperator> wave;
    HFWalker<CPS, Slater> walk;
    wave.appendSinglesToOpList(.0);
    wave.appendScreenedDoublesToOpList(.0);
    runVMC(wave, walk);
  }

  else if (schd.wavefunctionType == "CIJastrowSlater") {
    CIWavefunction<CPSSlater<Jastrow, Slater>, HFWalker<Jastrow, Slater>, SpinFreeOperator> wave;
    HFWalker<Jastrow, Slater> walk;
    wave.appendSinglesToOpList(.0);
    wave.appendScreenedDoublesToOpList(.0);
    runVMC(wave, walk);
  }
  
  else if (schd.wavefunctionType == "LanczosCPSSlater") {
    Lanczos<CPSSlater<CPS, Slater>> wave; HFWalker<CPS, Slater> walk;
    wave.initWalker(walk); 
    wave.optimizeWave(walk);
    wave.writeWave();
  }
  else if (schd.wavefunctionType == "LanczosJastrowSlater") {
    Lanczos<CPSSlater<Jastrow, Slater>> wave; HFWalker<Jastrow, Slater> walk;
    wave.initWalker(walk); 
    wave.optimizeWave(walk);
    wave.writeWave();

    double ham, stddev, rk;
    getEnergyDeterministic(wave, walk, ham);
    //getStochasticEnergyContinuousTime(wave, walk, ham, stddev, rk, schd.stochasticIter, 1.e-5);
    if (commrank == 0) cout << "Energy of VMC wavefunction: "<<ham <<"("<<stddev<<")"<<endl;
    
  }
  
  /*
  else if (schd.wavefunctionType == "CPSAGP") {
    CPSSlater<AGP, Slater> wave;
    HFWalker<AGP, Slater> walk;
    runVMC(wave, walk);
  }



    //vector<double> alpha{0., 0.1, 0.2, -0.1, -0.2}; 
    //vector<double> Ealpha{0., 0., 0., 0., 0.}; 
    //double stddev, rk;
    //for (int i = 0; i < alpha.size(); i++) {
    //  vars[0] = alpha[i];
    //  wave.updateVariables(vars);
    //  wave.initWalker(walk);
    //  getStochasticEnergyContinuousTime(wave, walk, Ealpha[i], stddev, rk, schd.stochasticIter, 1.e-5);
    //  if (commrank == 0) cout << alpha[i] << "   " << Ealpha[i] << "   " << stddev << endl;
    //}

    //getGradientWrapper<CIWavefunction<CPSSlater, HFWalker, Operator>, HFWalker> wrapper(wave, walk, schd.stochasticIter);
    //getGradientWrapper<Lanczos<CPSSlater, HFWalker>, HFWalker> wrapper(wave, walk, schd.stochasticIter);
    //  functor1 getStochasticGradient = boost::bind(&getGradientWrapper<Lanczos<CPSSlater, HFWalker>, HFWalker>::getGradient, &wrapper, _1, _2, _3, _4, _5, schd.deterministic);

    //if (schd.method == amsgrad) {
    //  AMSGrad optimizer(schd.stepsize, schd.decay1, schd.decay2, schd.maxIter);
    //  //functor1 getStochasticGradient = boost::bind(&getGradientWrapper<CIWavefunction<CPSSlater, HFWalker, Operator>, HFWalker>::getGradient, &wrapper, _1, _2, _3, _4, _5, schd.deterministic);
    //  optimizer.optimize(vars, getStochasticGradient, schd.restart);
    //  //if (commrank == 0) wave.printVariables();
    //}
    //else if (schd.method == sgd) {
    //  SGD optimizer(schd.stepsize, schd.maxIter);
    //  optimizer.optimize(vars, getStochasticGradient, schd.restart);
    //}
  }
  */
  
  boost::interprocess::shared_memory_object::remove(shciint2.c_str());
  boost::interprocess::shared_memory_object::remove(shciint2shm.c_str());
  return 0;
}
