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
#ifndef rHFWalker_HEADER_H
#define rHFWalker_HEADER_H

#include "rDeterminants.h"
#include "rWalkerHelper.h"
#include <array>
#include "igl/slice.h"
#include "igl/slice_into.h"
#include "Slater.h"

using namespace Eigen;

template<typename T1, typename T2>
struct rWalker {
};

template<>
struct rWalker<rJastrow, Slater> {

  rDeterminant d;
  rWalkerHelper<rJastrow> corrHelper;
  rWalkerHelper<Slater> refHelper;

  rWalker() {};
  
  rWalker(const rJastrow &corr, const Slater &ref) 
  {
    initDet(ref.getHforbsA(), ref.getHforbsB());
    refHelper = rWalkerHelper<Slater>(ref, d);
    corrHelper = rWalkerHelper<rJastrow>(corr, d);
  }

  rWalker(const rJastrow &corr, const Slater &ref, const rDeterminant &pd) : d(pd), refHelper(ref, pd), corrHelper(corr, pd) {}; 

  rDeterminant& getDet() {return d;}
  void readBestDeterminant(rDeterminant& d) const 
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


  double getDetOverlap(const Slater &ref) const
  {
    return refHelper.thetaDet[0][0]*refHelper.thetaDet[0][1];
  }

  /**
   * makes det based on mo coeffs 
   */
  void guessBestDeterminant(rDeterminant& d, const Eigen::MatrixXd& HforbsA, const Eigen::MatrixXd& HforbsB) const 
  {
    auto random = std::bind(std::uniform_real_distribution<double>(-1., 1.),
                            std::ref(generator));
    for (int i=0; i<d.nelec; i++) {
      d.coord[i][0] = random();
      d.coord[i][1] = random();
      d.coord[i][2] = random();
    }
  }

  void initDet(const MatrixXd& HforbsA, const MatrixXd& HforbsB) 
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


  void updateWalker(int elec, Vector3d& coord, const Slater& ref, const rJastrow& corr) {
    Vector3d oldCoord = d.coord[elec];
    d.coord[elec] = coord;
    corrHelper.updateWalker(elec, oldCoord, corr, d);
    refHelper.updateWalker(elec, oldCoord, d, ref);
  }

  void OverlapWithGradient(const Slater &ref, Eigen::VectorBlock<VectorXd> &grad) 
  {
    return;
    grad[0] = 0.0;
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

    int var = 1;
    for (int moa=0; moa<nalpha; moa++) {//alpha mo 
      for (int orb=0; orb<norbs; orb++) {//ao
        grad[var] = refHelper.thetaInv[0].row(moa).dot(AoRia.col(orb));
        var++;
      }
    }

    for (int mob=0; mob<nbeta; mob++) {//beta mo 
      for (int orb=0; orb<norbs; orb++) {//ao
        grad[var] = refHelper.thetaInv[1].row(mob).dot(AoRib.col(orb));
        var++;
      }
    }
  }
  

};


#endif
