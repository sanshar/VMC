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
#include <vector>
#include <algorithm>
#include "integral.h"
#include "Determinants.h"
#include <boost/format.hpp>
#include <iostream>
#include <fstream>
#include "global.h"
#include "input.h"

#ifndef SERIAL
#include "mpi.h"
#endif

using namespace std;
using namespace Eigen;

double calcTcorr(std::vector<double> &v);
void comb(int N, int K, std::vector<std::vector<int>> &combinations);


template <typename Wfn, typename Walker>
double evaluatePTDeterministic(Wfn& w, Walker& walk, double& E0,
                               double& stddev, double& rk, int niter,
                               double targetError) {

  int norbs = Determinant::norbs;
  int nalpha = Determinant::nalpha;
  int nbeta = Determinant::nbeta;

  vector<vector<int> > alphaDets, betaDets;
  comb(norbs, nalpha, alphaDets);
  comb(norbs, nbeta , betaDets);
  std::vector<Determinant> allDets;
  for (int a=0; a<alphaDets.size(); a++)
    for (int b=0; b<betaDets.size(); b++) {
      Determinant d;
      for (int i=0; i<alphaDets[a].size(); i++)
	d.setoccA(alphaDets[a][i], true);
      for (int i=0; i<betaDets[b].size(); i++)
	d.setoccB(betaDets[b][i], true);
      allDets.push_back(d);
    }

  alphaDets.clear(); betaDets.clear();
  workingArray work;
  work.nExcitations = 0;
  
  double A=0, B=0, C=0, ovlp=0;
  for (int d=commrank; d<allDets.size(); d+=commsize) {
    double Eloc=0, ovlploc=0; 
    double scale = 1.0;
    Walker walk;
    w.initWalker(walk, allDets[d]);

    double Ei = allDets[d].Energy(I1, I2, coreE);
    w.HamAndOvlp(walk, ovlploc, Eloc, work);
    work.nExcitations = 0;
  

    double ovlp2 = ovlploc*ovlploc;

    A    -= pow(Eloc-E0, 2)*ovlp2/(Ei-E0);
    B    += (Eloc-E0)*ovlp2/(Ei-E0);
    C    += ovlp2/(Ei-E0);
    ovlp += ovlp2;
  }
  allDets.clear();

  double obkp = ovlp;
  int size = 1;
#ifndef SERIAL
  MPI_Allreduce(&obkp, &ovlp,  size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
  double Abkp=A/ovlp;
  double Bbkp=B/ovlp, Cbkp = C/ovlp;

#ifndef SERIAL
  MPI_Allreduce(&Abkp, &A,     size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&Bbkp, &B,     size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&Cbkp, &C,     size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
  if (commrank == 0) cout <<A<<"  "<< B<<"  "<<C<<"  "<<B*B/C<<endl;
  return A + B*B/C;
}

template <typename Wfn, typename Walker>
double evaluatePTStochastic(Wfn &w, Walker& walk,
                            double& lambda, double &E0,
                            double& stddev, double& rk,
                            int niter, double targetError)
{
  int norbs = Determinant::norbs;
  int nalpha = Determinant::nalpha;
  int nbeta = Determinant::nbeta;

  auto random = std::bind(std::uniform_real_distribution<double>(0, 1),
                          std::ref(generator));

  workingArray work;
  work.nExcitations = 0;

  double Ept2;
  
  stddev = 1.e4;
  int iter = 0;
  double M1 = 0., S1 = 0., Eavg = 0.;
  double Aloc = 0.;
  double ham = 0., ovlp = 0.;
  double scale = 1.0;
  
  
  double bestOvlp =0.;
  Determinant bestDet=walk.getDet();
  
  
  double A = 0.0, B = 0.0, C = 0.0;
  w.HamAndOvlp(walk, ovlp, ham, work);

  int nstore = 100000/commsize;
  int gradIter = min(nstore, niter);

  std::vector<double> gradError(gradIter*commsize, 0);
  double cumdeltaT = 0., cumdeltaT2 = 0.;

  while (iter < niter && stddev > targetError)
  {
    double cumovlpRatio = 0;
    //when using uniform probability 1./numConnection * max(1, pi/pj)
    for (int i = 0; i < work.nExcitations; i++)
    {
      cumovlpRatio += abs(work.ovlpRatio[i]);
      work.ovlpRatio[i] = cumovlpRatio;
    }

    //double deltaT = -log(random())/(cumovlpRatio);
    double deltaT = 1.0 / (cumovlpRatio);
    double nextDetRandom = random()*cumovlpRatio;
    int nextDet = std::lower_bound(work.ovlpRatio.begin(),
                                   (work.ovlpRatio.begin()+work.nExcitations),
                                   nextDetRandom) - work.ovlpRatio.begin();
    

    cumdeltaT += deltaT;
    cumdeltaT2 += deltaT * deltaT;
    
    double Alocold = Aloc;
    
    double ratio = deltaT/cumdeltaT;

    double Ei = walk.d.Energy(I1, I2, coreE);
    Aloc = -pow(ham-E0, 2)/(Ei - E0);
    double Bloc = (ham-E0)/(Ei-E0);
    double Cloc = 1/(Ei-E0);
    A = A + deltaT * (Aloc - A) / (cumdeltaT);       //running average of energy
    B = B + deltaT * (Bloc - B) / (cumdeltaT);       //running average of energy
    C = C + deltaT * (Cloc - C) / (cumdeltaT);       //running average of energy
    S1 = S1 + (A - Alocold) * (A - Aloc);
    
    
    if (iter < gradIter)
      gradError[iter + commrank*gradIter] = Aloc;
    
    iter++;
    
    walk.updateWalker(w.getRef(), w.getCPS(), work.excitation1[nextDet], work.excitation2[nextDet]);

    work.nExcitations = 0;
    
    w.HamAndOvlp(walk, ovlp, ham, work);
  }
  
#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &(gradError[0]), gradError.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &A, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &B, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &C, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  rk = calcTcorr(gradError);

  A /= commsize;
  B /= commsize;
  C /= commsize;
  Ept2 = A + B*B/C;

  if (commrank == 0) cout << A<<"  "<<B*B/C<<endl;
  stddev = sqrt(S1 * rk / (niter - 1) / niter / commsize);
#ifndef SERIAL
  MPI_Bcast(&stddev, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif

  return Ept2;
}

