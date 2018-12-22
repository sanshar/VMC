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

using namespace boost;

rWalker<rJastrow, Slater>::rWalker(const rJastrow &corr, const Slater &ref) 
{
  initDet(ref.getHforbsA(), ref.getHforbsB());
  refHelper = rWalkerHelper<Slater>(ref, d);
  corrHelper = rWalkerHelper<rJastrow>(corr, d);
}

rWalker<rJastrow, Slater>::rWalker(const rJastrow &corr, const Slater &ref, const rDeterminant &pd) : d(pd), refHelper(ref, pd), corrHelper(corr, pd) {}; 

rDeterminant& rWalker<rJastrow, Slater>::getDet() {return d;}
void rWalker<rJastrow, Slater>::readBestDeterminant(rDeterminant& d) const 
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


double rWalker<rJastrow, Slater>::getDetOverlap(const Slater &ref) const
{
  return refHelper.thetaDet[0][0]*refHelper.thetaDet[0][1];
}

/**
 * makes det based on mo coeffs 
 */
void rWalker<rJastrow, Slater>::guessBestDeterminant(rDeterminant& d, const Eigen::MatrixXd& HforbsA, const Eigen::MatrixXd& HforbsB) const 
{
  auto random = std::bind(std::uniform_real_distribution<double>(-1., 1.),
                          std::ref(generator));
  for (int i=0; i<d.nelec; i++) {
    d.coord[i][0] = random();
    d.coord[i][1] = random();
    d.coord[i][2] = random();
  }
}

void rWalker<rJastrow, Slater>::initDet(const MatrixXd& HforbsA, const MatrixXd& HforbsB) 
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


void rWalker<rJastrow, Slater>::updateWalker(int elec, Vector3d& coord, const Slater& ref, const rJastrow& corr) {
  Vector3d oldCoord = d.coord[elec];
  d.coord[elec] = coord;
  corrHelper.updateWalker(elec, oldCoord, corr, d);
  refHelper.updateWalker(elec, oldCoord, d, ref);
}

void rWalker<rJastrow, Slater>::OverlapWithGradient(const Slater &ref, Eigen::VectorBlock<VectorXd> &grad) 
{
  grad[0] = 0.0;
  if (schd.optimizeOrbs == false) return;
  
  int norbs = schd.gBasis.norbs;
  int nalpha = rDeterminant::nalpha;
  int nbeta = rDeterminant::nbeta;
  int nelec = nalpha+nbeta;
  
  MatrixXd AoRia = MatrixXd::Zero(nalpha, norbs);
  MatrixXd AoRib = MatrixXd::Zero(nbeta, norbs);
  refHelper.aoValues.resize(norbs);
  
  for (int elec=0; elec<nalpha; elec++) {
    schd.gBasis.eval(d.coord[elec], &refHelper.aoValues[0]);
    for (int orb = 0; orb<norbs; orb++)
      AoRia(elec, orb) = refHelper.aoValues[orb];
  }
  
  for (int elec=0; elec<nbeta; elec++) {
    schd.gBasis.eval(d.coord[elec+nalpha], &refHelper.aoValues[0]);
    for (int orb = 0; orb<norbs; orb++)
      AoRib(elec, orb) = refHelper.aoValues[orb];
  }
  
  //Assuming a single determinant
  int numDets = ref.determinants.size();
  for (int moa=0; moa<nalpha; moa++) {//alpha mo 
    for (int orb=0; orb<norbs; orb++) {//ao
      grad[numDets + orb * norbs + moa] += refHelper.thetaInv[0].row(moa).dot(AoRia.col(orb));
    }
  }
  
  for (int mob=0; mob<nbeta; mob++) {//beta mo 
    for (int orb=0; orb<norbs; orb++) {//ao
      if (ref.hftype == Restricted) 
        grad[numDets + orb * norbs + mob] += refHelper.thetaInv[1].row(mob).dot(AoRib.col(orb));
      else
        grad[numDets + norbs*norbs + orb * norbs + mob] += refHelper.thetaInv[1].row(mob).dot(AoRib.col(orb));
    }
  }
}


void rWalker<rJastrow, Slater>::OverlapWithGradientGhf(const Slater &ref, Eigen::VectorBlock<VectorXd> &grad) 
{
  grad[0] = 0.0;
  int norbs = schd.gBasis.norbs;
  int nalpha = rDeterminant::nalpha;
  int nbeta = rDeterminant::nbeta;
  int nelec = nalpha+nbeta;
  
  MatrixXd AoRi = MatrixXd::Zero(nelec, 2*norbs);
  refHelper.aoValues.resize(norbs);
  
  for (int elec=0; elec<nelec; elec++) {
    schd.gBasis.eval(d.coord[elec], &refHelper.aoValues[0]);
    for (int orb = 0; orb<norbs; orb++) {
      if (elec < nalpha)
        AoRi(elec, orb) = refHelper.aoValues[orb];
      else
        AoRi(elec, norbs+orb) = refHelper.aoValues[orb];
    }
  }

  /*
  cout << refHelper.DetMatrix[0]<<endl<<endl;
  cout << refHelper.DetMatrix[0].determinant()<<endl;
  MatrixXd temp = refHelper.DetMatrix[0];
  temp.col(0) = AoRi.col(0);
  cout << temp <<endl;
  cout << temp.determinant()<<endl;
  cout << refHelper.thetaInv[0].row(0).dot(AoRi.col(0))<<endl;
  MatrixXd refinv = refHelper.DetMatrix[0].inverse();
  cout << refinv.row(0). dot(temp.col(0))<<endl;
  exit(0);
  */
  int numDets = ref.determinants.size();
  for (int mo=0; mo<nelec; mo++) {
    for (int orb=0; orb<2*norbs; orb++) {
      grad[numDets + orb * 2 * norbs + mo] += refHelper.thetaInv[0].row(mo).dot(AoRi.col(orb));
    }
  }
  
}
  



