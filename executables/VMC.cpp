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

#ifndef SERIAL
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif

#include "evaluateE.h"
#include "MoDeterminants.h"
#include "Determinants.h"
#include "CPS.h"
#include "Wfn.h"
#include "input.h"
#include "integral.h"
#include "SHCIshm.h"
#include "math.h"
#include "diis.h"
#include "optimizer.h"

using namespace Eigen;
using namespace boost;
using namespace std;


int main(int argc, char* argv[]) {

#ifndef SERIAL
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;
#endif
  startofCalc = getTime();

  initSHM();
  license();

  string inputFile = "input.dat";
  if (argc > 1)
    inputFile = string(argv[1]);
  if (commrank == 0) readInput(inputFile, schd);
#ifndef SERIAL
  mpi::broadcast(world, schd, 0);
#endif

  generator = std::mt19937(schd.seed+commrank);

  twoInt I2; oneInt I1;
  int norbs, nalpha, nbeta;
  double coreE=0.0;
  std::vector<int> irrep;
  readIntegrals("FCIDUMP", I2, I1, nalpha, nbeta, norbs, coreE, irrep);

  //initialize the heatbath integrals
  std::vector<int> allorbs;
  for (int i=0; i<norbs; i++)
    allorbs.push_back(i);
  twoIntHeatBath I2HB(1.e-10);
  twoIntHeatBathSHM I2HBSHM(1.e-10);
  if (commrank == 0) I2HB.constructClass(allorbs, I2, I1, norbs);
  I2HBSHM.constructClass(norbs, I2HB);


  //Setup static variables
  Determinant::EffDetLen = (norbs)/64+1;
  Determinant::norbs     = norbs;
  MoDeterminant::norbs   = norbs;
  MoDeterminant::nalpha  = nalpha;
  MoDeterminant::nbeta   = nbeta;

  //Setup Slater Determinants
  Hforbs = MatrixXd::Zero(norbs, norbs);
  readHF(Hforbs);
  MatrixXd alpha(norbs, nalpha), beta(norbs, nbeta);
  alpha = Hforbs.block(0, 0, norbs, nalpha);
  beta  = Hforbs.block(0, 0, norbs, nbeta );
  MoDeterminant det(alpha, beta);


  //Setup CPS wavefunctions
  std::vector<Correlator> nSiteCPS;
  for (auto it = schd.correlatorFiles.begin(); it != schd.correlatorFiles.end();
       it++) {
    readCorrelator(it->second, it->first, nSiteCPS);
  }

  vector<Determinant> detList(1); vector<double> ciExpansion(1, 1.0);
  for (int i=0; i<nalpha; i++)
    detList[0].setoccA(i, true);
  for (int i=0; i<nbeta; i++)
    detList[0].setoccB(i, true);

  //setup up wavefunction
  CPSSlater wave(nSiteCPS, detList, ciExpansion);//(nSiteCPS, det); //******** CHANGE THIS
  if (schd.restart) {
    ifstream file ("params_min.bin", ios::in|ios::binary|ios::ate);
    size_t size = file.tellg();
    Eigen::VectorXd vars = Eigen::VectorXd::Zero(size/sizeof(double));
    file.seekg (0, ios::beg);
    file.read ( (char*)(&vars[0]), size);
    file.close();

    if (vars.size() != wave.getNumVariables()) {
      cout << "number of variables on disk: "<<vars.size()<<" is not equal to wfn parameters: "<<wave.getNumVariables()<<endl;
      exit(0);
    }

    wave.updateVariables(vars);

    if (commrank == 0) cout << "Calculating the energy of the wavefunction."<<endl;
    double stddev=0.;
    double E = evaluateEStochastic(wave, nalpha, nbeta, norbs, I1, I2, I2HBSHM, coreE, stddev, schd.stochasticIter, 0.5e-3);
    if (commrank == 0) cout << format("%14.8f (%8.2e)\n") %(E) % (stddev);
  }
  else {
    cout << "Use the python script to optimize the wavefunction."<<endl;
    exit(0);
  }
  cout.precision(10);


  //optimize the wavefunction
  //if (schd.m == rmsprop)
  //optimizer::rmsprop(wave, I1, I2, coreE);


  boost::interprocess::shared_memory_object::remove(shciint2.c_str());
  return 0;
}