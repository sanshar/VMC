/*
  Developed by Sandeep Sharma with contributions from James E. T. Smith and Adam A. Holmes, 2017
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

#ifndef SERIAL
#include "mpi.h"
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif
#include "rWalker.h"
#include "math.h"
#include "rCorrelatedWavefunction.h"

using namespace boost;

rWalker<rJastrow, rSlater>::rWalker() {
  uR = std::bind(std::uniform_real_distribution<double>(0, 1),
                 std::ref(generator));
  nR = std::normal_distribution<double>(.0,1.0); //0 mean and 1 stddev  
}

rWalker<rJastrow, rSlater>::rWalker(const rJastrow &corr, const rSlater &ref) 
{
  uR = std::bind(std::uniform_real_distribution<double>(0, 1),
                 std::ref(generator));
  nR = std::normal_distribution<double> (.0,1.0); //0 mean and 1 stddev  

  initDet(ref.getHforbsA(), ref.getHforbsB());

  initR();
  initHelpers(corr, ref);
}

void rWalker<rJastrow, rSlater>::initHelpers(const rJastrow &corr, const rSlater &ref)  {
  refHelper = rWalkerHelper<rSlater>(ref, d);
  corrHelper = rWalkerHelper<rJastrow>(corr, d, Rij, RiN);
}

void rWalker<rJastrow, rSlater>::initR() {
  Rij = MatrixXd::Zero(d.nelec, d.nelec);
  for (int i=0; i<d.nelec; i++)
    for (int j=0; j<i; j++) {
      double rij = pow( pow(d.coord[i][0] - d.coord[j][0], 2) +
                        pow(d.coord[i][1] - d.coord[j][1], 2) +
                        pow(d.coord[i][2] - d.coord[j][2], 2), 0.5);

      Rij(i,j) = rij;
      Rij(j,i) = rij;        
    }

  RiN = MatrixXd::Zero(d.nelec, schd.Ncoords.size());
  for (int i=0; i<d.nelec; i++)
    for (int j=0; j<schd.Ncoords.size(); j++) {
      double rij = pow( pow(d.coord[i][0] - schd.Ncoords[j][0], 2) +
                        pow(d.coord[i][1] - schd.Ncoords[j][1], 2) +
                        pow(d.coord[i][2] - schd.Ncoords[j][2], 2), 0.5);

      RiN(i,j) = rij;
    }
    
}

//rWalker<rJastrow, rSlater>::rWalker(const rJastrow &corr, const rSlater &ref, const rDeterminant &pd) : d(pd), refHelper(ref, pd), corrHelper(corr, pd) {}; 

rDeterminant& rWalker<rJastrow, rSlater>::getDet() {return d;}
void rWalker<rJastrow, rSlater>::readBestDeterminant(rDeterminant& d) const 
{
  if (commrank == 0) {
    char file[5000];
    sprintf(file, "BestCoordinates.txt");
    std::ifstream ifs(file, std::ios::binary);
    boost::archive::binary_iarchive load(ifs);
    load >> d;
  }
#ifndef SERIAL
  boost::mpi::communicator world;
  mpi::broadcast(world, d, 0);
#endif
}


double rWalker<rJastrow, rSlater>::getDetOverlap(const rSlater &ref) const
{
  return refHelper.thetaDet[0][0]*refHelper.thetaDet[0][1];
}

/**
 * makes det based on mo coeffs 
 */
void rWalker<rJastrow, rSlater>::guessBestDeterminant(rDeterminant& d, const Eigen::MatrixXd& HforbsA, const Eigen::MatrixXd& HforbsB) const 
{
  auto random = std::bind(std::uniform_real_distribution<double>(-1., 1.),
                          std::ref(generator));
  for (int i=0; i<d.nelec; i++) {
    d.coord[i][0] = random();
    d.coord[i][1] = random();
    d.coord[i][2] = random();
  }
}

void rWalker<rJastrow, rSlater>::initDet(const MatrixXd& HforbsA, const MatrixXd& HforbsB) 
{
  bool readDeterminant = false;
  char file[5000];
  sprintf(file, "BestCoordinates.txt");
  
  {
    ifstream ofile(file);
    if (ofile)
      readDeterminant = true;
  }
  if (readDeterminant)
    readBestDeterminant(d);
  else
    guessBestDeterminant(d, HforbsA, HforbsB);
}


void rWalker<rJastrow, rSlater>::updateWalker(int elec, Vector3d& coord, const rSlater& ref, const rJastrow& corr) {
  Vector3d oldCoord = d.coord[elec];
  d.coord[elec] = coord;
  for (int j=0; j<d.nelec; j++) {
    Rij(elec, j) = pow( pow(d.coord[elec][0] - d.coord[j][0], 2) +
                     pow(d.coord[elec][1] - d.coord[j][1], 2) +
                     pow(d.coord[elec][2] - d.coord[j][2], 2), 0.5);

    Rij(j,elec) = Rij(elec,j);
  }

  for (int j=0; j<schd.Ncoords.size(); j++) {
    RiN(elec, j) = pow( pow(d.coord[elec][0] - schd.Ncoords[j][0], 2) +
                     pow(d.coord[elec][1] - schd.Ncoords[j][1], 2) +
                     pow(d.coord[elec][2] - schd.Ncoords[j][2], 2), 0.5);
  }

  corrHelper.updateWalker(elec, oldCoord, corr, d, Rij, RiN);
  refHelper.updateWalker(elec, oldCoord, d, ref);

}

void rWalker<rJastrow, rSlater>::OverlapWithGradient(const rSlater &ref,
                                                    const rJastrow& cps,
                                                    VectorXd &grad) 
{
  double factor1 = 1.0;
  corrHelper.OverlapWithGradient(cps, grad, d, factor1);
  
  Eigen::VectorBlock<VectorXd> gradtail = grad.tail(grad.rows() - cps.getNumVariables());
  if (schd.optimizeOrbs == false) return;
  refHelper.OverlapWithGradient(d, ref, gradtail, factor1);
}


void rWalker<rJastrow, rSlater>::HamOverlap(const rSlater &ref,
                                           const rJastrow& cps,
                                           VectorXd &hamgrad) 
{
  //double factor1 = 1.0;
  //corrHelper.HamOverlap(cps, grad, d, factor1);
  
  Eigen::VectorBlock<VectorXd> hamtail = hamgrad.tail(hamgrad.rows()
                                                      - cps.getNumVariables());

  if (schd.optimizeOrbs == false) return;
  refHelper.HamOverlap(d, ref, Rij, RiN, hamtail);
}


void rWalker<rJastrow, rSlater>::getSimpleStep(Vector3d& coord, double stepsize) {
  coord[0] = (uR()-0.5)*stepsize;
  coord[1] = (uR()-0.5)*stepsize;
  coord[2] = (uR()-0.5)*stepsize;
}




