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
#include "rDeterminants.h"
#include "rJastrow.h"
#include "rSlater.h"
#include "rBFSlater.h"
#include "input.h"
#include "SHCIshm.h"
#include "math.h"
#include "Profile.h"
#include "rCorrelatedWavefunction.h"
#include "propagate.h"

using namespace Eigen;
using namespace boost;
using namespace std;

//functions to test cusp conditions
template<typename Wfn, typename Walker>
double testElectronCusp(Wfn &wave, Walker &walk)
{
  Vector3d r0;
  r0 << 0.5, 0, 0;
  walk.updateWalker(0, r0, wave.getRef(), wave.getCorr());
  cout << "initial coordinates" << endl;
  cout << walk.d << endl;
  for (double theta = -M_PI / 4; theta <= M_PI / 4; theta += 0.005)
  {
    Vector3d r1;
    r1 << std::cos(theta), std::sin(theta), 0;
    r1 *= r0.norm();
    walk.updateWalker(1, r1, wave.getRef(), wave.getCorr());
    double ovlp = wave.Overlap(walk);
    double Eloc = wave.rHam(walk);
    std::cout << std::fixed;
    std::cout << std::setprecision(6);
    cout << theta << " " << ovlp << " " << Eloc << endl;
  }
  return 0;
}


template<typename Wfn, typename Walker>
double testNuclearCusp(Wfn &wave, Walker &walk)
{
  Vector3d r0;
  r0 << 0, 0, 1.0;
  walk.updateWalker(0, r0, wave.getRef(), wave.getCorr());
  const Vector3d& N0 = schd.Ncoords[0];
  //const Vector3d& N1 = schd.Ncoords[1];
  cout << "Atom 1: " << N0.transpose();
  //cout << "Atom 2: " << N1.transpose() << endl;
  cout << walk.RiN << endl;
  cout << "initial coordinates" << endl;
  cout << walk.d << endl;
  for (double z = 0.2; z >= -0.2; z -= 0.005)
  {
    Vector3d r1;
    r1 << 0, 0, z;
    //r1 << std::cos(theta), std::sin(theta), 0;
    //r1 *= r0.norm();
    walk.updateWalker(1, r1, wave.getRef(), wave.getCorr());
    double ovlp = wave.Overlap(walk);
    double Eloc = wave.rHam(walk);
    std::cout << std::fixed;
    std::cout << std::setprecision(6);
    cout << z << " " << ovlp << " " << Eloc << endl;
  }
  return 0;
}


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
    //initialize wavefunction
    rCorrelatedWavefunction<rJastrow, rSlater> wave; rWalker<rJastrow, rSlater> walk;
    wave.readWave(); wave.initWalker(walk);

    /*
    if (commrank == 0)
    {
        cout << "EEcusp" << endl;
        testElectronCusp(wave, walk);
        cout << endl;
        cout << "ENcusp" << endl;
        testNuclearCusp(wave, walk);
        cout << endl;
    }
    */

    //calculate the energy as a initial guess for shift
    double E0, error, rk;
    double acceptedFrac = getEnergyMetropolisRealSpace(wave, walk, E0, error, rk, schd.stochasticIter, 5.0e-3);
    if (commrank == 0) cout << "Energy of VMC wavefunction: " << E0 << " ("<< error << ")" << endl;


    //do DMC
    doDMC(wave, walk, E0);
  } 

  boost::interprocess::shared_memory_object::remove(shciint2.c_str());
  boost::interprocess::shared_memory_object::remove(shciint2shm.c_str());
  boost::interprocess::shared_memory_object::remove(shciint2shmcas.c_str());
  return 0;
}

