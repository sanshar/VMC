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

#ifndef RJastrow_HEADER_H
#define RJastrow_HEADER_H
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <string>
#include "JastrowTerms.h"

/*
 * Jastrow is a product of correlators 
 */
class rJastrow {
 private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize (Archive & ar, const unsigned int version) {
    ar & _jastrow & _params & _gradHelper;
  }
 public:

  GeneralJastrow _jastrow;
  std::vector<double> _params; 
  std::vector<double> _gradHelper; 


  rJastrow ();
  
  double exponentialInitLaplaceGrad(const rDeterminant& d,
                                    MatrixXd& Gradient,
                                    VectorXd& laplacian,
                                    vector<MatrixXd>& paramgradient,
                                    MatrixXd& paramlaplacian) ;
  
  double exponentDiff(int i, Vector3d& coord, const rDeterminant& d) ;

  void UpdateLaplaceGrad(MatrixXd& Gradient,
                         VectorXd& laplacian,
                         vector<MatrixXd>& paramgradient,
                         MatrixXd& paramlaplacian,
                         const MatrixXd& Rij,
                         const MatrixXd& RiN,
                         const rDeterminant& d,
                         const Vector3d& oldCoord, int i) const ;
  
  void OverlapWithGradient( VectorXd& grad, const rDeterminant& d) const;
  long getNumVariables() const;

  void getVariables(Eigen::VectorXd &v) const;
  void updateVariables(const Eigen::VectorXd &v);
  void printVariables() const;
  std::string getfileName() const {return "rJastrow";};
};


#endif
