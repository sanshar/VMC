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
#include "input.h"
#include "integral.h"
#include "SHCIshm.h"
#include "math.h"
#include "Profile.h"
#include "CIWavefunction.h"
#include "CorrelatedWavefunction.h"
#include "Lanczos.h"
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

  cout.precision(10);
  string inputFile = "input.dat";
  if (argc > 1)
    inputFile = string(argv[1]);
  readInput(inputFile, schd, false);

  generator = std::mt19937(schd.seed + commrank);

  if (schd.walkerBasis == ORBITALS) {
    readIntegralsAndInitializeDeterminantStaticVariables("FCIDUMP");
    
    
    //calculate the hessian/gradient
    if (schd.wavefunctionType == "CPSSlater") {
      CorrelatedWavefunction<CPS, Slater> wave; Walker<CPS, Slater> walk;
      runVMC(wave, walk);
    }
    
    else if (schd.wavefunctionType == "CPSAGP") {
      CorrelatedWavefunction<CPS, AGP> wave; Walker<CPS, AGP> walk;
      runVMC(wave, walk);
    }
    
    else if (schd.wavefunctionType == "CPSPfaffian") {
      CorrelatedWavefunction<CPS, Pfaffian> wave; Walker<CPS, Pfaffian> walk;
      runVMC(wave, walk);
    }
    
    if (schd.wavefunctionType == "JastrowSlater") {
      CorrelatedWavefunction<Jastrow, Slater> wave; Walker<Jastrow, Slater> walk;
      runVMC(wave, walk);
    }
    
    else if (schd.wavefunctionType == "JastrowAGP") {
      CorrelatedWavefunction<Jastrow, AGP> wave; Walker<Jastrow, AGP> walk;
      runVMC(wave, walk);
    }
    
    else if (schd.wavefunctionType == "JastrowPfaffian") {
      CorrelatedWavefunction<Jastrow, Pfaffian> wave; Walker<Jastrow, Pfaffian> walk;
      runVMC(wave, walk);
    }
    
    else if (schd.wavefunctionType == "CICPSSlater") {
      CIWavefunction<CorrelatedWavefunction<CPS, Slater>, Walker<CPS, Slater>, SpinFreeOperator> wave; Walker<CPS, Slater> walk;
      wave.appendSinglesToOpList(0.0); wave.appendScreenedDoublesToOpList(0.0);
      runVMC(wave, walk);
    }
    
    else if (schd.wavefunctionType == "CICPSAGP") {
      CIWavefunction<CorrelatedWavefunction<CPS, AGP>, Walker<CPS, AGP>, SpinFreeOperator> wave; Walker<CPS, AGP> walk;
      wave.appendSinglesToOpList(0.0); wave.appendScreenedDoublesToOpList(0.0);
      runVMC(wave, walk);
    }
    
    else if (schd.wavefunctionType == "CICPSPfaffian") {
      CIWavefunction<CorrelatedWavefunction<CPS, Pfaffian>, Walker<CPS, Pfaffian>, SpinFreeOperator> wave; Walker<CPS, Pfaffian> walk;
      wave.appendSinglesToOpList(0.0); wave.appendScreenedDoublesToOpList(0.0);
      runVMC(wave, walk);
    }
    
    else if (schd.wavefunctionType == "CIJastrowSlater") {
      CIWavefunction<CorrelatedWavefunction<Jastrow, Slater>, Walker<Jastrow, Slater>, SpinFreeOperator> wave; Walker<Jastrow, Slater> walk;
      wave.appendSinglesToOpList(0.0); wave.appendScreenedDoublesToOpList(0.0);
      runVMC(wave, walk);
    }
    
    else if (schd.wavefunctionType == "CIJastrowAGP") {
      CIWavefunction<CorrelatedWavefunction<Jastrow, AGP>, Walker<Jastrow, AGP>, SpinFreeOperator> wave; Walker<Jastrow, AGP> walk;
      wave.appendSinglesToOpList(0.0); wave.appendScreenedDoublesToOpList(0.0);
      runVMC(wave, walk);
    }
    
    else if (schd.wavefunctionType == "CIJastrowPfaffian") {
      CIWavefunction<CorrelatedWavefunction<Jastrow, Pfaffian>, Walker<Jastrow, Pfaffian>, SpinFreeOperator> wave; Walker<Jastrow, Pfaffian> walk;
      wave.appendSinglesToOpList(0.0); wave.appendScreenedDoublesToOpList(0.0);
      runVMC(wave, walk);
    }
    
    else if (schd.wavefunctionType == "LanczosCPSSlater") {
      Lanczos<CorrelatedWavefunction<CPS, Slater>> wave; Walker<CPS, Slater> walk;
      wave.initWalker(walk);
      wave.optimizeWave(walk);
      wave.writeWave();
    }
    
    else if (schd.wavefunctionType == "LanczosCPSAGP") {
      Lanczos<CorrelatedWavefunction<CPS, AGP>> wave; Walker<CPS, AGP> walk;
      wave.initWalker(walk);
      wave.optimizeWave(walk);
      wave.writeWave();
    }
    
    else if (schd.wavefunctionType == "LanczosCPSPfaffian") {
      Lanczos<CorrelatedWavefunction<CPS, Pfaffian>> wave; Walker<CPS, Pfaffian> walk;
      wave.initWalker(walk);
      wave.optimizeWave(walk);
      wave.writeWave();
    }
    
    else if (schd.wavefunctionType == "LanczosJastrowSlater") {
      Lanczos<CorrelatedWavefunction<Jastrow, Slater>> wave; Walker<Jastrow, Slater> walk;
      wave.initWalker(walk);
      wave.optimizeWave(walk);
      wave.writeWave();
    }
    
    else if (schd.wavefunctionType == "LanczosJastrowAGP") {
      Lanczos<CorrelatedWavefunction<Jastrow, AGP>> wave; Walker<Jastrow, AGP> walk;
      wave.initWalker(walk);
      wave.optimizeWave(walk);
      wave.writeWave();
    }
    
    else if (schd.wavefunctionType == "LanczosJastrowPfaffian") {
      Lanczos<CorrelatedWavefunction<Jastrow, Pfaffian>> wave; Walker<Jastrow, Pfaffian> walk;
      wave.initWalker(walk);
      wave.optimizeWave(walk);
      wave.writeWave();
    }
  }
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
      
    if (schd.wavefunctionType == "JastrowSlater") {
      rCorrelatedWavefunction<rJastrow, rSlater> wave; rWalker<rJastrow, rSlater> walk;
      runVMCRealSpace(wave, walk);
    }
  }    
  

  boost::interprocess::shared_memory_object::remove(shciint2.c_str());
  boost::interprocess::shared_memory_object::remove(shciint2shm.c_str());
  return 0;
}
