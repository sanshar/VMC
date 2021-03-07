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
//#define EIGEN_USE_MKL_ALL
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
#include "mpi.h"
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif
#include <boost/function.hpp>
#include <boost/functional.hpp>
#include <boost/bind.hpp>
#include "evaluateE.h"
#include "Determinants.h"
#include "rDeterminants.h"
#include "rJastrow.h"
#include "rSlater.h"
#include "rMultiSlater.h"
#include "rBFSlater.h"
#include "input.h"
#include "integral.h"
#include "SHCIshm.h"
#include "math.h"
#include "Profile.h"
#include "CIWavefunction.h"
#include "CorrelatedWavefunction.h"
#include "ResonatingWavefunction.h"
#include "ResonatingTRWavefunction.h"
#include "TRWavefunction.h"
#include "PermutedWavefunction.h"
#include "PermutedTRWavefunction.h"
#include "SelectedCI.h"
#include "SimpleWalker.h"
#include "Lanczos.h"
#include "SCCI.h"
#include "SCPT.h"
#include "MRCI.h"
#include "EOM.h"
#include "runVMC.h"
#include "rCorrelatedWavefunction.h"

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
  if (commrank == 0) {
    system("echo User:; echo $USER");
    system("echo Hostname:; echo $HOSTNAME");
    system("echo CPU info:; lscpu | head -15");
    system("echo Computation started at:; date");
    cout << "git commit: " << GIT_HASH << ", branch: " << GIT_BRANCH << ", compiled at: " << COMPILE_TIME << endl << endl;
    cout << "nproc used: " << commsize << " (NB: stochasticIter below is per proc)" << endl << endl; 
  }

  cout.precision(10);
  string inputFile = "input.dat";
  if (argc > 1)
    inputFile = string(argv[1]);
  readInput(inputFile, schd, true);

  generator = std::mt19937(schd.seed + commrank);

  if (schd.walkerBasis == ORBITALS) {
    readIntegralsAndInitializeDeterminantStaticVariables("FCIDUMP");
    if (schd.numActive == -1) schd.numActive = Determinant::norbs;
    
    if (schd.wavefunctionType == "jastrowslater") {
      CorrelatedWavefunction<Jastrow, Slater> wave; Walker<Jastrow, Slater> walk;
      runVMC(wave, walk);
    }
      
    else if (schd.wavefunctionType == "gutzwillerslater") {
      CorrelatedWavefunction<Gutzwiller, Slater> wave; Walker<Gutzwiller, Slater> walk;
      runVMC(wave, walk);
    }

    else if (schd.wavefunctionType == "cpsslater") {
      CorrelatedWavefunction<CPS, Slater> wave; Walker<CPS, Slater> walk;
      runVMC(wave, walk);
    }
    
    else if (schd.wavefunctionType == "cpsagp") {
      CorrelatedWavefunction<CPS, AGP> wave; Walker<CPS, AGP> walk;
      runVMC(wave, walk);
    }
    
    else if (schd.wavefunctionType == "cpspfaffian") {
      CorrelatedWavefunction<CPS, Pfaffian> wave; Walker<CPS, Pfaffian> walk;
      runVMC(wave, walk);
    }
    
    
    else if (schd.wavefunctionType == "resonatingwavefunction") {
      ResonatingWavefunction wave; ResonatingWalker walk;
      runVMC(wave, walk);
    }
    
    else if (schd.wavefunctionType == "resonatingtrwavefunction") {
      ResonatingTRWavefunction wave; ResonatingTRWalker walk;
      runVMC(wave, walk);
    }
    
    else if (schd.wavefunctionType == "trwavefunction") {
      TRWavefunction wave; TRWalker walk;
      runVMC(wave, walk);
    }
    
    else if (schd.wavefunctionType == "permutedwavefunction") {
      PermutedWavefunction wave; PermutedWalker walk;
      runVMC(wave, walk);
    }
    
    else if (schd.wavefunctionType == "permutedtrwavefunction") {
      PermutedTRWavefunction wave; PermutedTRWalker walk;
      runVMC(wave, walk);
    }
    
    
    else if (schd.wavefunctionType == "jastrowagp") {
      CorrelatedWavefunction<Jastrow, AGP> wave; Walker<Jastrow, AGP> walk;
      runVMC(wave, walk);
    }
    
    else if (schd.wavefunctionType == "jastrowpfaffian") {
      CorrelatedWavefunction<Jastrow, Pfaffian> wave; Walker<Jastrow, Pfaffian> walk;
      runVMC(wave, walk);
    }
    
    else if (schd.wavefunctionType == "rbm") {
      CorrelatedWavefunction<RBM, Slater> wave; Walker<RBM, Slater> walk;
      runVMC(wave, walk);
    }
    
    else if (schd.wavefunctionType == "jrbms") {
      CorrelatedWavefunction<JRBM, Slater> wave; Walker<JRBM, Slater> walk;
      runVMC(wave, walk);
    }
    
    else if (schd.wavefunctionType == "jrbmp") {
      CorrelatedWavefunction<JRBM, Pfaffian> wave; Walker<JRBM, Pfaffian> walk;
      runVMC(wave, walk);
    }
    
    else if (schd.wavefunctionType == "cicpsslater") {
      CIWavefunction<CorrelatedWavefunction<CPS, Slater>, Walker<CPS, Slater>, SpinFreeOperator> wave; Walker<CPS, Slater> walk;
      wave.appendSinglesToOpList(0.0); wave.appendScreenedDoublesToOpList(0.0);
      runVMC(wave, walk);
    }
    
    else if (schd.wavefunctionType == "cicpsagp") {
      CIWavefunction<CorrelatedWavefunction<CPS, AGP>, Walker<CPS, AGP>, SpinFreeOperator> wave; Walker<CPS, AGP> walk;
      wave.appendSinglesToOpList(0.0); wave.appendScreenedDoublesToOpList(0.0);
      runVMC(wave, walk);
    }
    
    else if (schd.wavefunctionType == "cicpspfaffian") {
      CIWavefunction<CorrelatedWavefunction<CPS, Pfaffian>, Walker<CPS, Pfaffian>, SpinFreeOperator> wave; Walker<CPS, Pfaffian> walk;
      wave.appendSinglesToOpList(0.0); wave.appendScreenedDoublesToOpList(0.0);
      runVMC(wave, walk);
    }
    
    else if (schd.wavefunctionType == "cijastrowslater") {
      //CIWavefunction<CorrelatedWavefunction<Jastrow, Slater>, Walker<Jastrow, Slater>, SpinFreeOperator> wave; Walker<Jastrow, Slater> walk;
      CIWavefunction<CorrelatedWavefunction<Jastrow, Slater>, Walker<Jastrow, Slater>, Operator> wave; Walker<Jastrow, Slater> walk;
      wave.appendSinglesToOpList(0.0); wave.appendScreenedDoublesToOpList(0.0);
      runVMC(wave, walk);
    }
    
    else if (schd.wavefunctionType == "cijastrowagp") {
      CIWavefunction<CorrelatedWavefunction<Jastrow, AGP>, Walker<Jastrow, AGP>, SpinFreeOperator> wave; Walker<Jastrow, AGP> walk;
      wave.appendSinglesToOpList(0.0); wave.appendScreenedDoublesToOpList(0.0);
      runVMC(wave, walk);
    }
    
    else if (schd.wavefunctionType == "cijastrowpfaffian") {
      CIWavefunction<CorrelatedWavefunction<Jastrow, Pfaffian>, Walker<Jastrow, Pfaffian>, SpinFreeOperator> wave; Walker<Jastrow, Pfaffian> walk;
      wave.appendSinglesToOpList(0.0); wave.appendScreenedDoublesToOpList(0.0);
      runVMC(wave, walk);
    }
    
    else if (schd.wavefunctionType == "sci") {
      CIWavefunction<SelectedCI, SimpleWalker, Operator> wave; SimpleWalker walk;
      wave.appendSinglesToOpList(0.0); wave.appendScreenedDoublesToOpList(0.0);
      runVMC(wave, walk);
    }
    
    else if (schd.wavefunctionType == "lanczoscpslater") {
      Lanczos<CorrelatedWavefunction<CPS, Slater>> wave; Walker<CPS, Slater> walk;
      wave.initWalker(walk);
      wave.optimizeWave(walk);
      wave.writeWave();
    }
    
    else if (schd.wavefunctionType == "lanczoscpsagp") {
      Lanczos<CorrelatedWavefunction<CPS, AGP>> wave; Walker<CPS, AGP> walk;
      wave.initWalker(walk);
      wave.optimizeWave(walk);
      wave.writeWave();
    }
    
    else if (schd.wavefunctionType == "lanczoscpspfaffian") {
      Lanczos<CorrelatedWavefunction<CPS, Pfaffian>> wave; Walker<CPS, Pfaffian> walk;
      wave.initWalker(walk);
      wave.optimizeWave(walk);
      wave.writeWave();
    }
    
    else if (schd.wavefunctionType == "lanczosjastrowslater") {
      Lanczos<CorrelatedWavefunction<Jastrow, Slater>> wave; Walker<Jastrow, Slater> walk;
      wave.initWalker(walk);
      wave.optimizeWave(walk);
      wave.writeWave();
    }
    
    else if (schd.wavefunctionType == "lanczosjastrowagp") {
      Lanczos<CorrelatedWavefunction<Jastrow, AGP>> wave; Walker<Jastrow, AGP> walk;
      wave.initWalker(walk);
      wave.optimizeWave(walk);
      wave.writeWave();
    }
    
    else if (schd.wavefunctionType == "lanczosjastrowpfaffian") {
      Lanczos<CorrelatedWavefunction<Jastrow, Pfaffian>> wave; Walker<Jastrow, Pfaffian> walk;
      wave.initWalker(walk);
      wave.optimizeWave(walk);
      wave.writeWave();
    }
    
    else if (schd.wavefunctionType == "lanczossci") {
      Lanczos<SelectedCI> wave; SimpleWalker walk;
      wave.initWalker(walk);
      double alpha = wave.optimizeWave(walk, schd.alpha);
      wave.writeWave();
    }
    
    else if (schd.wavefunctionType == "scci") {
      SCCI<SelectedCI> wave; SimpleWalker walk;
      wave.initWalker(walk);
      if (schd.method == linearmethod) {
        wave.optimizeWaveCTDirect(walk); 
        wave.optimizeWaveCTDirect(walk);
      }
      else {
        runVMC(wave, walk);
        wave.calcEnergy(walk);
      }
      wave.writeWave();
    }
    
    else if (schd.wavefunctionType == "MRCI") {
      MRCI<Jastrow, Slater> wave; MRCIWalker<Jastrow, Slater> walk;
      //wave.initWalker(walk);
      runVMC(wave, walk);
    }
    
    else if (schd.wavefunctionType == "eom") {
      EOM<Jastrow, Slater> wave; Walker<Jastrow, Slater> walk;
      //cout << "detetrministic\n\n";
      if (schd.deterministic) wave.optimizeWaveDeterministic(walk); 
      else {
        wave.initWalker(walk);
        wave.optimizeWaveCT(walk); 
      }
      //wave.calcPolDeterministic(walk); 
      //wave.calcPolCT(walk); 
      //wave.initWalker(walk);
      //runVMC(wave, walk);
    }
    
    else if (schd.wavefunctionType == "pol") {
      EOM<Jastrow, Slater> wave; Walker<Jastrow, Slater> walk;
      //cout << "detetrministic\n\n";
      //wave.optimizeWaveDeterministic(walk); 
      //cout << "stochastic\n\n";
      //wave.optimizeWaveCT(walk); 
      if (schd.deterministic) wave.calcPolDeterministic(walk); 
      else {
        wave.initWalker(walk);
        wave.calcPolCT(walk); 
      }
      //wave.initWalker(walk);
      //runVMC(wave, walk);
    }
    
    else if (schd.wavefunctionType == "scpt") {
      SCPT<SelectedCI> wave; SimpleWalker walk;
      wave.initWalker(walk);
      wave.optimizeWaveCT(walk);
      wave.optimizeWaveCT(walk);
    }

    else if (schd.wavefunctionType == "slaterrdm") {
      CorrelatedWavefunction<Jastrow, Slater> wave; Walker<Jastrow, Slater> walk;
      wave.readWave();
      MatrixXd oneRdm0, oneRdm1, corr;
      //getOneRdmDeterministic(wave, walk, oneRdm0, 0);
      //getOneRdmDeterministic(wave, walk, oneRdm1, 1);
      //getDensityCorrelationsDeterministic(wave, walk, corr);
      getStochasticOneRdmContinuousTime(wave, walk, oneRdm0, 0, schd.stochasticIter);
      getStochasticOneRdmContinuousTime(wave, walk, oneRdm1, 1, schd.stochasticIter);
      getStochasticDensityCorrelationsContinuousTime(wave, walk, corr, schd.stochasticIter);
      if (commrank == 0) {
        cout << "oneRdm0\n" << oneRdm0 << endl << endl;
        cout << "oneRdm1\n" << oneRdm1 << endl << endl;
        cout << "spat "<<oneRdm0+oneRdm1<<endl<<endl;
        cout << "Density correlations\n" << corr << endl << endl;
      }
    }
    
    else if (schd.wavefunctionType == "slatertwordm") {
      CorrelatedWavefunction<Jastrow, Slater> wave; Walker<Jastrow, Slater> walk;
      wave.readWave();
      MatrixXd twoRdm;
      //getOneRdmDeterministic(wave, walk, oneRdm0, 0);
      //getOneRdmDeterministic(wave, walk, oneRdm1, 1);
      //getDensityCorrelationsDeterministic(wave, walk, corr);
      if (schd.deterministic)
        getTwoRdmDeterministic(wave, walk, twoRdm);
      else
        getStochasticTwoRdmContinuousTime(wave, walk, twoRdm, schd.stochasticIter);
    }
    
    else if (schd.wavefunctionType == "agprdm") {
      CorrelatedWavefunction<Jastrow, AGP> wave; Walker<Jastrow, AGP> walk;
      wave.readWave();
      MatrixXd oneRdm0, oneRdm1, corr;
      //getOneRdmDeterministic(wave, walk, oneRdm0, 0);
      //getOneRdmDeterministic(wave, walk, oneRdm1, 1);
      //getDensityCorrelationsDeterministic(wave, walk, corr);
      getStochasticOneRdmContinuousTime(wave, walk, oneRdm0, 0, schd.stochasticIter);
      getStochasticOneRdmContinuousTime(wave, walk, oneRdm1, 1, schd.stochasticIter);
      getStochasticDensityCorrelationsContinuousTime(wave, walk, corr, schd.stochasticIter);
      if (commrank == 0) {
        cout << "oneRdm0\n" << oneRdm0 << endl << endl;
        cout << "oneRdm1\n" << oneRdm1 << endl << endl;
        cout << "Density correlations\n" << corr << endl << endl;
      }
    }
    
    else if (schd.wavefunctionType == "pfaffianrdm") {
      CorrelatedWavefunction<Jastrow, Pfaffian> wave; Walker<Jastrow, Pfaffian> walk;
      wave.readWave();
      MatrixXd oneRdm0, oneRdm1, corr;
      //getOneRdmDeterministic(wave, walk, oneRdm0, 0);
      //getOneRdmDeterministic(wave, walk, oneRdm1, 1);
      //getDensityCorrelationsDeterministic(wave, walk, corr);
      getStochasticOneRdmContinuousTime(wave, walk, oneRdm0, 0, schd.stochasticIter);
      getStochasticOneRdmContinuousTime(wave, walk, oneRdm1, 1, schd.stochasticIter);
      getStochasticDensityCorrelationsContinuousTime(wave, walk, corr, schd.stochasticIter);
      if (commrank == 0) {
        cout << "oneRdm0\n" << oneRdm0 << endl << endl;
        cout << "oneRdm1\n" << oneRdm1 << endl << endl;
        cout << "Density correlations\n" << corr << endl << endl;
      }
    }
  }//orbital space
  
  else {//real space VMC
    rDeterminant::nalpha = schd.nalpha;
    rDeterminant::nbeta = schd.nbeta;
    Determinant::nalpha = schd.nalpha;
    Determinant::nbeta = schd.nbeta;
    rDeterminant::nelec = schd.nalpha + schd.nbeta;
    //Determinant::nelec = schd.nalpha + schd.nbeta;
    Determinant::norbs = schd.norbs;
    if (schd.nalpha< 0 ||
        schd.nbeta < 0 ||
        schd.nalpha+schd.nbeta <= 0) {
      cout << "need to supply nalpha and nbeta electrons in the input."<<endl;
      exit(0);
    }
    if (schd.wavefunctionType == "jastrowslater") {
      rCorrelatedWavefunction<rJastrow, rSlater> wave; rWalker<rJastrow, rSlater> walk;
      runVMCRealSpace(wave, walk);
    }

    else if (schd.wavefunctionType == "jastrowmultislater") {
      rCorrelatedWavefunction<rJastrow, rMultiSlater> wave; rWalker<rJastrow, rMultiSlater> walk;
      runVMCRealSpace(wave, walk);
    }
    
    else if (schd.wavefunctionType == "backflow") {
      rCorrelatedWavefunction<rJastrow, rBFSlater> wave; rWalker<rJastrow, rBFSlater> walk;
      runVMCRealSpace(wave, walk);
    }
  }
      

  boost::interprocess::shared_memory_object::remove(shciint2.c_str());
  boost::interprocess::shared_memory_object::remove(shciint2shm.c_str());
  boost::interprocess::shared_memory_object::remove(shciint2shmcas.c_str());
  return 0;
}
