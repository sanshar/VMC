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
#include <ctime>
#include <sys/time.h>
#include <algorithm>
#include <random>
#include "time.h"
#include <chrono>
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <boost/format.hpp>
#ifndef SERIAL
//#include "mpi.h"
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

using namespace Eigen;
using namespace boost;
using namespace std;

double getTime() {
  struct timeval start;
  gettimeofday(&start, NULL);
  return start.tv_sec + 1.e-6*start.tv_usec;
}
double startofCalc = getTime();


void license() {
  if (commrank == 0) {
  cout << endl;
  cout << endl;
  cout << "**************************************************************"<<endl;
  cout << "Dice  Copyright (C) 2017  Sandeep Sharma"<<endl;
  cout <<"This program is distributed in the hope that it will be useful,"<<endl;
  cout <<"but WITHOUT ANY WARRANTY; without even the implied warranty of"<<endl;
  cout <<"MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE."<<endl;  
  cout <<"See the GNU General Public License for more details."<<endl;
  cout << endl<<endl;
  cout << "Author:       Sandeep Sharma"<<endl;
  cout << "Please visit our group page for up to date information on other projects"<<endl;
  cout << "http://www.colorado.edu/lab/sharmagroup/"<<endl;
  cout << "**************************************************************"<<endl;
  cout << endl;
  cout << endl;
  }
}



int main(int argc, char* argv[]) {

#ifndef SERIAL
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;
#endif

  license();
  initSHM();

  twoInt I2; oneInt I1; 
  int norbs, nalpha, nbeta; 
  double coreE=0.0;
  std::vector<int> irrep;
  readIntegrals("FCIDUMP", I2, I1, nalpha, nbeta, norbs, coreE, irrep);
  
  Determinant::EffDetLen = (norbs*2)/64+1;
  Determinant::norbs    = norbs*2;
  HalfDet::norbs        = norbs;
  MoDeterminant::norbs  = norbs;
  MoDeterminant::nalpha = nalpha;
  MoDeterminant::nbeta  = nbeta;

  MatrixXd Hforbs = MatrixXd::Zero(norbs, norbs);
  readHF(Hforbs);

  MatrixXd alpha(norbs, nalpha), beta(norbs, nbeta);
  alpha = Hforbs.block(0, 0, norbs, nalpha);
  beta  = Hforbs.block(0, 0, norbs, nbeta );

  MoDeterminant det(alpha, beta);

  std::vector<CPS> twoSiteCPS;
  //aa
  for (int i=0; i<norbs; i++)
  for (int j=i; j<norbs; j++) {
    if (j == i)
    {
      vector<int> asites(1,i), bsites(1,j);
      twoSiteCPS.push_back(CPS(asites, bsites));
    }
    /*
    if (j >= i)
    {
      vector<int> asites(1,i), bsites(1,j);
      twoSiteCPS.push_back(CPS(asites, bsites));
    }
    if (j >i ) {
      vector<int> asites(2,0), bsites;
      asites[0] = i; asites[1] = j;
      twoSiteCPS.push_back(CPS(asites, bsites));
      twoSiteCPS.push_back(CPS(bsites, asites));
    }
    */
  }


  CPSSlater wave(twoSiteCPS, det);

  Determinant initial;
  for (int i=0; i<nalpha; i++) initial.setocc(2*i  , true);
  for (int i=0; i<nbeta;  i++) initial.setocc(2*i+1, true);



  for (int iter =0; iter<500; iter++) {
    double E0 = evaluateEDeterministic(wave, nalpha, nbeta, norbs, I1, I2, coreE);
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(wave.getNumVariables());
    getGradient(wave, E0, nalpha, nbeta, norbs, I1, I2, coreE, grad);
    cout << iter<<"  "<<E0<<"  "<<grad.norm()<<endl;
    //wave.printVariables();
    grad *= -0.005;
    wave.updateVariables(grad);
  }

  exit(0);
  /*
  auto random = std::bind(std::uniform_real_distribution<double>(0,1),
			  mt19937(getTime()+commrank));

  int iter = 0, newiter=0; double cumulative=0.0;
  double ovlp, ham;
  det.HamAndOvlp(alphaOrbs, betaOrbs, ovlp, ham, I1, I2, coreE);
  bool update = true;

  while (iter < 100000) {

    double Eloc = ham/ovlp; 

    cumulative += Eloc;
    iter ++;
    if (iter %1000 == 0) {
      double cum = cumulative;
      int cumiter = iter*commsize;
      MPI_Allreduce(&cumulative, &cum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      if (commrank == 0)
	std::cout << format("%6i   %14.8f  %14.8f  %14.8f \n") %iter % ovlp % Eloc % (cum/cumiter);
    }

    //pick a random occupied orbital
    int i = floor( random()*(alphaOrbs.size()+betaOrbs.size()) );
    if (i < alphaOrbs.size()) {
      int a = floor(random()*alphaOpen.size());
      std::swap(alphaOrbs[i], alphaOpen[a]);

      double newovlp = det.Overlap(alphaOrbs, betaOrbs);
      if (pow(newovlp/ovlp,2) > random() ) {
	newiter++;
	det.HamAndOvlp(alphaOrbs, betaOrbs, ovlp, ham, I1, I2, coreE);
      }
      else 
	std::swap(alphaOrbs[i], alphaOpen[a]);
    }
    else {
      i = i - alphaOrbs.size();
      int a = floor( random()*betaOpen.size());
      std::swap(betaOrbs[i], betaOpen[a]);

      double newovlp = det.Overlap(alphaOrbs, betaOrbs);
      if (pow(newovlp/ovlp,2) > random() ) {
	det.HamAndOvlp(alphaOrbs, betaOrbs, ovlp, ham, I1, I2, coreE);
	newiter++;
      }
      else 
	std::swap(betaOrbs[i], betaOpen[a]);

    }

  }
  */
  boost::interprocess::shared_memory_object::remove(shciint2.c_str());
  return 0;
}
