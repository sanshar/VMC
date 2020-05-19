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

#ifndef relDeterminants_HEADER_H
#define relDeterminants_HEADER_H

#include "global.h"
#include <iostream>
#include <vector>
#include <boost/serialization/serialization.hpp>
#include <Eigen/Dense>

class oneInt;
class twoInt;

class oneIntSOC;

class twoIntHeatBathSHM;
class relWorkingArray;
class workingArray;

using namespace std;



class relDeterminant {

 private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    for (int i=0; i<DetLen; i++)
      ar & reprA[i] & reprB[i];
  }

 public:
  
  // 0th position of 0th long is the first position
  // 63rd position of the last long is the last position
  long reprA[DetLen], reprB[DetLen];
  static int EffDetLen;
  static char Trev;
  static int norbs;
  static int nalpha, nbeta;
  static Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic> LexicalOrder;
  

  //Constructors
  relDeterminant();
  relDeterminant(const relDeterminant& d);
  void operator=(const relDeterminant& d);

  
  //mutates the Determinant
  void setoccA(int i, bool occ);
  void setoccB(int i, bool occ);
  void setocc(int i, bool occ) ; //i is the spin orbital index
  void setocc(int i, bool sz, bool occ) ; //i is the spatial orbital index, sz=0 for alpha, =1 for beta
  
  bool getocc(int i) const ;//i is the spin orbital index
  bool getocc(int i, bool sz) const ;//i is the spatial orbital index 
  bool getoccA(int i) const ;
  bool getoccB(int i) const;
  void getOpenClosed( std::vector<int>& open, std::vector<int>& closed) const;//gets spin orbitals
  void getOpenClosed( bool sz, std::vector<int>& open, std::vector<int>& closed) const;//gets spatial orbitals
  void getOpenClosedAlphaBeta( std::vector<int>& openAlpha,
                               std::vector<int>& closedAlpha,
                               std::vector<int>& openBeta,
                               std::vector<int>& closedBeta ) const;
  void getClosedAlphaBeta( std::vector<int>& closedAlpha,
                           std::vector<int>& closedBeta ) const;
  void getAlphaBeta(std::vector<int>& alpha, std::vector<int>& beta) const;
  void getClosed(bool sz, std::vector<int>& closed) const;
  int getNbetaBefore(int i) const;
  int getNalphaBefore(int i) const;
  int Noccupied() const;
  int Nalpha() const;
  int Nbeta() const;

  double parityA(const int& a, const int& i) const;
  double parityB(const int& a, const int& i) const;
  double parity(const int& a, const int& i, const bool& sz) const;
  double parityA(const vector<int>& aArray, const vector<int>& iArray) const ;
  double parityB(const vector<int>& aArray, const vector<int>& iArray) const ;
  double parity(const vector<int>& aArray, const vector<int>& iArray, bool sz) const ;
  double parityAA(const int& i, const int& j, const int& a, const int& b) const ;
  double parityBB(const int& i, const int& j, const int& a, const int& b) const ;

  double relParity(const int& a, const int& i) const ;
  double relParity_old(const int& a, const int& i) const ;
  double relParityProbablyWrong(const int& a, const int& i) const ;
  double relParityAsInDice(const int& a, const int& i) const ;

  std::complex<double> Energy(const oneIntSOC& I1, const twoInt& I2, const double& coreE) const ;
  std::complex<double> Hij_1ExciteScreened(const int& a, const int& i, const twoIntHeatBathSHM& Ishm,
                             const double& TINY, bool doparity = true) const;
  /*
  std::complex<double> Hij_1ExciteA(const int& a, const int& i, const oneInt&I1,  const twoInt& I2,
                      bool doparity=true) const ;
  std::complex<double> Hij_1ExciteB(const int& a, const int& i, const oneInt&I1, const twoInt& I2,
                      bool doparity=true) const ;
  std::complex<double> Hij_2ExciteAA(const int& a, const int& i, const int& b, const int& j,
                       const oneInt&I1, const twoInt& I2) const ;
  std::complex<double> Hij_2ExciteBB(const int& a, const int& i, const int& b, const int& j,
                       const oneInt&I1, const twoInt& I2) const ;
  std::complex<double> Hij_2ExciteAB(const int& a, const int& i, const int& b, const int& j,
                       const oneInt&I1, const twoInt& I2) const ;

  */
  bool connected(const relDeterminant& d) const;
  int ExcitationDistance(const relDeterminant& d) const;


  //operators
  bool operator<(const relDeterminant& d) const;
  bool operator==(const relDeterminant& d) const;
  friend ostream& operator<<(ostream& os, const relDeterminant& d);
 
};


/*
//note some of i, j, k, l might be repeats
//and some its possible that the determinant might get killed
//the return value tell us whether the determinant is killed
bool applyExcitation(int a, int b, int k, int l, Determinant& dcopy);

CItype Hij(const Determinant& bra, const Determinant& ket,
           const oneInt& I1, const twoInt& I2, const double& coreE);

void getDifferenceInOccupation(const Determinant &bra, const Determinant &ket,
                               vector<int> &creA, vector<int> &desA,
                               vector<int> &creB, vector<int> &desB);
void getDifferenceInOccupation(const Determinant &bra, const Determinant &ket,
                               vector<int> &cre, vector<int> &des,
                               bool sz);
void getDifferenceInOccupation(const Determinant &bra, const Determinant &ket, int &I, int &A);
void getDifferenceInOccupation(const Determinant &bra, const Determinant &ket,
                               int &I, int &J, int& A, int& B);

*/
double getParityForDiceToAlphaBeta(const relDeterminant& det);

/*
void generateScreenedSingleDoubleExcitation(const Determinant& d,
                                             const double& THRESH,
                                             const double& TINY,
                                             relWorkingArray& work,
                                             bool doparity);


*/
void generateAllScreenedDoubleExcitation(const relDeterminant& det,
                                         const double& screen,
                                         const double& TINY,
                                         relWorkingArray& work,
                                         bool doparity = false);

void generateAllScreenedDoubleExcitation(const relDeterminant& det,
                                         const double& screen,
                                         const double& TINY,
                                         workingArray& work,
                                         bool doparity = false);



void generateAllScreenedSingleExcitation(const relDeterminant& det,
                                         const double& screen,
                                         const double& TINY,
                                         relWorkingArray& work,
                                         bool doparity = false);

void generateNoSpinFlipScreenedSingleExcitation(const relDeterminant& det,
                                                const double& screen,
                                                const double& TINY,
                                                relWorkingArray& work,
                                                bool doparity = false);

void generateAllScreenedSingleExcitation(const relDeterminant& det,
                                         const double& screen,
                                         const double& TINY,
                                         workingArray& work,
                                         bool doparity = false);

void generateAllSingleExcitation(const relDeterminant& det,
                                 relWorkingArray& work,
                                 bool doparity = false);



void generateAllDeterminants(vector<relDeterminant>& allDets, int norbs, int nalpha, int nbeta);


void relGenerateAllDeterminants(vector<relDeterminant>& allDets, int norbs, int nelec);


void relComb(int N, int K, vector<vector<int>> &combinations);
/*


void readrelDeterminants(std::string input, vector<relDeterminant> &determinants,
                      vector<double> &ciExpansion);
*/

#endif
