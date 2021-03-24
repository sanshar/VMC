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
#ifndef RMultiSlater_HEADER_H
#define RMultiSlater_HEADER_H
#include <vector>
#include <set>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <Eigen/Dense>
#include "Determinants.h" 
#include "rMultiSlater.h"
#include "global.h"
#include "input.h"

class oneInt;
class twoInt;
class twoIntHeatBathSHM;
class workingArray;


/**
 * This is the wavefunction, it is a linear combination of
 * slater determinants made of Hforbs
 */
class rMultiSlater {
 private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & ref
       & open
       //& ciExcitations
       //& ciIndices
       & ciParity
       & ciCoeffs
       & numDets
       & HforbsA
       & HforbsB;
  }
  
  
 public:
 
  std::array<std::vector<int>, 2> ref;                                      // reference determinant electron occupations
  std::array<std::vector<int>, 2> open;                                      // reference determinant hole occupations
  std::array<std::vector<std::array<Eigen::VectorXi, 2>>, 2> ciExcitations; // ci expansion excitations
  std::array<std::vector<std::array<Eigen::VectorXi, 2>>, 2> ciIndices; // ci expansion excitation internal indices
  std::vector<double> ciParity;                                 // parity factors for the ci exctiations
  std::vector<double> ciCoeffs;                              // ci coeffs
  size_t numDets;                                            // ci expansion size
  Eigen::MatrixXcd HforbsA;                                 //alpha mo coeff
  Eigen::MatrixXcd HforbsB;                                 //beta mo coeff 

  //read mo coeffs from hf.txt
  void initHforbs();

  //initialize the ci expansion by reading determinants generated from Dice
  void initCiExpansion();

  //constructor
  rMultiSlater();

  //variables are ordered as:
  //cicoeffs of the reference multidet expansion, followed by hforbs (row major): real and complex parts alternating
  size_t getNumOfDets() const { return numDets; }

  long getNumVariables() const
  {
    int norbs = Determinant::norbs;
    int nact = norbs;
    if (schd.nciAct > 0) { nact = schd.nciAct; }
    return numDets + norbs * nact;
  }

  const Eigen::MatrixXcd &getHforbs(int sz) const
  { 
    if (sz == 0) { return HforbsA; }
    else { return HforbsB; }
  }

  void getVariables(Eigen::VectorBlock<Eigen::VectorXd> &v) const
  {
    int norbs = Determinant::norbs;
    int nact = norbs;
    if (schd.nciAct > 0) { nact = schd.nciAct; }

    for (size_t i = 0; i < numDets; i++) { v[i] = ciCoeffs[i]; }

    for (int p = 0; p < norbs; p++)
    {
      for (int q = 0; q < nact; q++)
      {
        v(numDets + p * nact + q) = HforbsA(p, q).real();
      }
    }
  }

  void getCiVariables(Eigen::VectorBlock<Eigen::VectorXd> &v) const
  {
    for (size_t i = 0; i < numDets; i++) { v[i] = ciCoeffs[i]; }
  }

  void getOrbVariables(Eigen::VectorBlock<Eigen::VectorXd> &v) const
  {
    int norbs = Determinant::norbs;
    int nact = norbs;
    if (schd.nciAct > 0) { nact = schd.nciAct; }

    for (int p = 0; p < norbs; p++)
    {
      for (int q = 0; q < nact; q++)
      {
        v(p * nact + q) = HforbsA(p, q).real();
      }
    }
  }

  void updateVariables(const Eigen::VectorBlock<Eigen::VectorXd> &v)
  {
    int norbs = Determinant::norbs;
    int nact = norbs;
    if (schd.nciAct > 0) { nact = schd.nciAct; }

    for (size_t i = 0; i < numDets; i++) { ciCoeffs[i] = v[i]; }

    for (int p = 0; p < norbs; p++)
    {
      for (int q = 0; q < nact; q++)
      {
        HforbsA(p, q) = std::complex<double>(v[numDets + p * nact + q], 0.0);
        HforbsB(p, q) = HforbsA(p, q);
      }
    }
  }

  void updateCiVariables(const Eigen::VectorBlock<Eigen::VectorXd> &v)
  {
    for (size_t i = 0; i < numDets; i++) { ciCoeffs[i] = v[i]; }
  }

  void updateOrbVariables(const Eigen::VectorBlock<Eigen::VectorXd> &v)
  {
    int norbs = Determinant::norbs;
    int nact = norbs;
    if (schd.nciAct > 0) { nact = schd.nciAct; }

    for (int p = 0; p < norbs; p++)
    {
      for (int q = 0; q < nact; q++)
      {
        HforbsA(p, q) = std::complex<double>(v[p * nact + q], 0.0);
        HforbsB(p, q) = HforbsA(p, q);
      }
    }
  }

  void printVariables() const;

  string getfileName() const { return "rMultiSlater"; }
};


#endif
