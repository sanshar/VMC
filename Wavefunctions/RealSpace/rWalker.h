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
  MatrixXd Rij;         //the inter-electron distances
  MatrixXd RiN;         //electron-nucleus distances  
  rWalkerHelper<rJastrow> corrHelper;
  rWalkerHelper<Slater> refHelper;

  rWalker() {};
  
  rWalker(const rJastrow &corr, const Slater &ref) ;

  //rWalker(const rJastrow &corr, const Slater &ref, const rDeterminant &pd);

  rDeterminant& getDet();
  void readBestDeterminant(rDeterminant& d) const ;


  double getDetOverlap(const Slater &ref) const;

  /**
   * makes det based on mo coeffs 
   */
  void guessBestDeterminant(rDeterminant& d, const Eigen::MatrixXd& HforbsA, const Eigen::MatrixXd& HforbsB) const ;

  void initDet(const MatrixXd& HforbsA, const MatrixXd& HforbsB) ;


  void updateWalker(int elec, Vector3d& coord, const Slater& ref, const rJastrow& corr);

  void OverlapWithGradient(const Slater &ref, const rJastrow& cps, VectorXd &grad) ;
  

};


#endif
