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
#include <iostream>
#include <algorithm>
#include "integral.h"
#include <Eigen/Dense>
#include <Eigen/Core>
#include "Determinants.h"

using namespace std;
using namespace Eigen;




//=============================================================================
double Determinant::Energy(oneInt& I1, twoInt&I2, double& coreE) {
	/*!
	   Calculates the energy of the determinant.

	   :Arguments:

  	   oneInt& I1:
  	       One body integrals.
  	   twoInt& I2:
  	       Two body integrals.
  	   double& coreE:
  	       Core energy.

	   :Returns:

  	   double energy+coreE:
  	       Energy of determinant.
	 */

	double energy = 0.0;
	size_t one = 1;
	vector<int> closed;
	for(int i=0; i<DetLen; i++) {
	  long reprBit = reprA[i];
	  while (reprBit != 0) {
	    int pos = __builtin_ffsl(reprBit);
	    closed.push_back( 2*(i*64+pos-1));
	    reprBit &= ~(one<<(pos-1));
	  }

	  reprBit = reprB[i];
	  while (reprBit != 0) {
	    int pos = __builtin_ffsl(reprBit);
	    closed.push_back( 2*(i*64+pos-1)+1);
	    reprBit &= ~(one<<(pos-1));
	  }
	}

	for (int i=0; i<closed.size(); i++) {
	  int I = closed.at(i);
#ifdef Complex
	  energy += I1(I,I).real();
#else
	  energy += I1(I,I);
#endif
	  for (int j=i+1; j<closed.size(); j++) {
	    int J = closed.at(j);
	    energy += I2.Direct(I/2,J/2);
	    if ( (I%2) == (J%2) ) {
	      energy -= I2.Exchange(I/2, J/2);
	    }
	  }
	}
	
	return energy+coreE;
}





//=============================================================================
void Determinant::parityAA(int& i, int& j, int& a, int& b, double& sgn) {
	parityA(a, i, sgn);
	setoccA(i, false); setoccA(a,true);
	parityA(b, j, sgn);
	setoccA(i, true); setoccA(a, false);
	return;
}

void Determinant::parityBB(int& i, int& j, int& a, int& b, double& sgn) {
	parityB(a, i, sgn);
	setoccB(i, false); setoccB(a,true);
	parityB(b, j, sgn);
	setoccB(i, true); setoccB(a, false);
	return;
}



//=============================================================================
CItype Determinant::Hij_2ExciteAA(int& a, int& i, int& b, int& j, oneInt&I1,
  twoInt& I2) {

  double sgn = 1.0;
  parityAA(i, j, a, b, sgn);
  return sgn*(I2(2*a,2*i,2*b,2*j) - I2(2*a,2*j,2*b,2*i));
}

CItype Determinant::Hij_2ExciteBB(int& a, int& i, int& b, int& j, oneInt&I1,
  twoInt& I2) {

  double sgn = 1.0;
  parityBB(i, j, a, b, sgn);
  return sgn*(I2(2*a+1, 2*i+1, 2*b+1, 2*j+1) - I2(2*a+1, 2*j+1, 2*b+1, 2*i+1 ));
}

CItype Determinant::Hij_2ExciteAB(int& a, int& i, int& b, int& j, oneInt&I1,
  twoInt& I2) {

  double sgn = 1.0;
  parityA(a, i, sgn); parityB(b,j,sgn); 
  return sgn*I2(2*a,2*i,2*b+1,2*j+1);
}



//=============================================================================
CItype Determinant::Hij_1ExciteA(int& a, int& i, oneInt&I1, twoInt& I2, bool doparity) {
  double sgn = 1.0;
  if (doparity) parityA(a, i, sgn);

  CItype energy = I1(2*a, 2*i);
  long one = 1;
  for (int I=0; I<DetLen; I++) {
    long reprBit = reprA[I];
    while (reprBit != 0) {
      int pos = __builtin_ffsl(reprBit);
      int j = I*64+pos-1;
      energy += (I2(2*a, 2*i, 2*j, 2*j) - I2(2*a, 2*j, 2*j, 2*i));
      reprBit &= ~(one<<(pos-1));
    }
    reprBit = reprB[I];
    while (reprBit != 0) {
      int pos = __builtin_ffsl(reprBit);
      int j = I*64+pos-1;
      energy += (I2(2*a, 2*i, 2*j+1, 2*j+1));
      reprBit &= ~(one<<(pos-1));
    }
    
  }
  energy *= sgn;
  return energy;
}

CItype Determinant::Hij_1ExciteB(int& a, int& i, oneInt&I1, twoInt& I2, bool doparity) {
  double sgn = 1.0;
  if (doparity) parityB(a, i, sgn);

  CItype energy = I1(2*a+1, 2*i+1);
  long one = 1;
  for (int I=0; I<DetLen; I++) {
    long reprBit = reprA[I];
    while (reprBit != 0) {
      int pos = __builtin_ffsl(reprBit);
      int j = I*64+pos-1;
      energy += (I2(2*a+1, 2*i+1, 2*j, 2*j));
      reprBit &= ~(one<<(pos-1));
    }
    reprBit = reprB[I];
    while (reprBit != 0) {
      int pos = __builtin_ffsl(reprBit);
      int j = I*64+pos-1;
      energy += (I2(2*a+1, 2*i+1, 2*j+1, 2*j+1) - I2(2*a+1, 2*j+1, 2*j+1, 2*i+1));
      reprBit &= ~(one<<(pos-1));
    }
    
  }
  energy *= sgn;
  return energy;
}



//=============================================================================
CItype Hij(Determinant& bra, Determinant& ket, oneInt& I1, twoInt& I2,
  double& coreE) {
    /*!
    Calculates the hamiltonian matrix element connecting the two determinants bra and ket.

    :Arguments:

        Determinant& bra:
            Determinant in bra.
        Determinant& ket:
            Determinant in ket.
        oneInt& I1:
           One body integrals.
        twoInt& I2:
           Two body integrals.
        double& coreE:
           Core energy.
        size_t& orbDiff:
            Number of orbitals with differing occupations.
    */
  int cre[200],des[200],ncrea=0,ncreb=0,ndesa=0,ndesb=0;
  long u,b,k,one=1;
  cre[0]=-1; cre[1]=-1; des[0]=-1; des[1]=-1;
  
  for (int i=0; i<Determinant::EffDetLen; i++) {
    u = bra.reprA[i] ^ ket.reprA[i];
    b = u & bra.reprA[i]; //the cre bits
    k = u & ket.reprA[i]; //the des bits
    
    while(b != 0) {
      int pos = __builtin_ffsl(b);
      cre[ncrea+ncreb] = 2*(pos-1+i*64);
      ncrea++;
      b &= ~(one<<(pos-1));
    }
    while(k != 0) {
      int pos = __builtin_ffsl(k);
      des[ndesa+ndesb] = 2*(pos-1+i*64);
      ndesa++;
      k &= ~(one<<(pos-1));
    }

    u = bra.reprB[i] ^ ket.reprB[i];
    b = u & bra.reprB[i]; //the cre bits
    k = u & ket.reprB[i]; //the des bits
    
    while(b != 0) {
      int pos = __builtin_ffsl(b);
      cre[ncrea+ncreb] = 2*(pos-1+i*64)+1;
      ncreb++;
      b &= ~(one<<(pos-1));
    }
    while(k != 0) {
      int pos = __builtin_ffsl(k);
      des[ndesa+ndesb] = 2*(pos-1+i*64)+1;
      ndesb++;
      k &= ~(one<<(pos-1));
    }

  }
  
  if (ncrea+ncreb == 0) {
    cout << bra<<endl;
    cout << ket<<endl;
    cout <<"Use the function for energy"<<endl;
    exit(0);
  }
  else if (ncrea == 1 && ncreb == 0) {
    int c0=cre[0]/2, d0 = des[0]/2;
    return ket.Hij_1ExciteA(c0, d0, I1, I2);
  }
  else if (ncrea == 0 && ncreb == 1) {
    int c0=cre[0]/2, d0 = des[0]/2;
    return ket.Hij_1ExciteB(c0, d0, I1, I2);
  }
  else if (ncrea == 0 && ncreb == 2) {
    int c0=cre[0]/2, d0 = des[0]/2;
    int c1=cre[1]/2, d1 = des[1]/2;
    return ket.Hij_2ExciteBB(c0, d0, c1, d1, I1, I2);
  }
  else if (ncrea == 2 && ncreb == 0) {
    int c0=cre[0]/2, d0 = des[0]/2;
    int c1=cre[1]/2, d1 = des[1]/2;
    return ket.Hij_2ExciteAA(c0, d0, c1, d1, I1, I2);
  }
  else if (ncrea == 1 && ncreb == 1) {
    int c0=cre[0]/2, d0 = des[0]/2;
    int c1=cre[1]/2, d1 = des[1]/2;
    if (cre[0]%2 == 0)
      return ket.Hij_2ExciteAB(c0, d0, c1, d1, I1, I2);
    else
      return ket.Hij_2ExciteAB(c1, d1, c0, d0, I1, I2);
  }
  else {
    return 0.;
  }
}


int Determinant::numberPossibleSingles(double& screen, oneInt& I1, twoInt& I2,
				       twoIntHeatBathSHM& I2hb) {
  double TINY = screen;
  vector<int> closed;
  vector<int> open;
  getOpenClosed(open, closed);
  
  int numSingles = 0;
  for (int i=0; i<closed.size(); i++) {
    for (int a=0; a<open.size(); a++) {
      if (closed[i]%2 == open[a]%2 && I2hb.Singles(closed[i], open[a]) > TINY) {
	int I = closed[i]/2, A = open[a]/2;
	double tia = 0;
	bool Alpha = closed[i]%2 == 0 ? true : false;
	if (Alpha) tia = Hij_1ExciteA(A, I, I1, I2);
	else Hij_1ExciteB(A, I, I1, I2);
	
	double localham = 0.0;
	if (abs(tia) > TINY) 
	  numSingles++;
      }
    }
  }
  return numSingles;
}


void sampleSingleDoubleExcitation(Determinant& d,  oneInt& I1, twoInt& I2, twoIntHeatBathSHM& I2hb,
				  int nsingles, int ndoubles,
				  vector<int>& Isingle, vector<int>& Asingle, 
				  vector<int>& Idouble, vector<int>& Adouble,
				  vector<int>& Jdouble, vector<int>& Bdouble, 
				  vector<double>& psingle, vector<double>& pdouble) {

  assert(nsingles != 0);
  assert(ndoubles != 0);

  auto random = std::bind(std::uniform_real_distribution<double>(0,1),
			  std::ref(generator));

  int norbs = Determinant::norbs;
  vector<int> closed;
  vector<int> open;
  closed.reserve(norbs); open.reserve(norbs);
  d.getOpenClosed(open, closed);

  vector<double> upperBoundOfSingles; //the maximum value of a single excitation h_ia
  vector<size_t> orbitalPairs;        //store all the orbital pairs i and a
  double         cumSingles = 0.0;    //sum of all the maximum excitations

  upperBoundOfSingles.reserve( closed.size()*open.size());
  orbitalPairs       .reserve( closed.size()*open.size());

  //generate a single excitation
  for (int i=0; i<closed.size(); i++)
  for (int a=0; a<open.size()  ; a++) {
    if (closed[i]%2 == open[a]%2) {
      int I = closed[i],
	  A = open[a];
      upperBoundOfSingles.push_back( cumSingles + I2hb.Singles(I, A));
      orbitalPairs       .push_back( I * 2 * norbs + A);
      cumSingles += I2hb.Singles(I, A);

    }
  }

  for (int i=0; i<nsingles; i++) {
    double selectSingle = random()*cumSingles;
    int    singleIndex  = std::lower_bound(upperBoundOfSingles.begin(), upperBoundOfSingles.end(),
					   selectSingle) - upperBoundOfSingles.begin();

    Isingle.push_back(orbitalPairs[singleIndex] / (2*norbs));
    Asingle.push_back(orbitalPairs[singleIndex] - Isingle[i] * 2 * norbs);

    //probability of having selected this single
    double psingletemp = singleIndex == 0 ? 
         upperBoundOfSingles[singleIndex]/cumSingles 
      : (upperBoundOfSingles[singleIndex] - upperBoundOfSingles[singleIndex-1])/cumSingles; 
    psingle.push_back(psingletemp);
  }

  //********************************
  //select a pair of occupied orbitals
  vector<double>    occPairProbability;
  vector<size_t>    occPair;
  double cumOccPair = 0.0;

  occPairProbability.reserve( closed.size()*closed.size());
  occPair           .reserve( closed.size()*closed.size());

  for (int i=0  ; i<closed.size(); i++)
  for (int j=0  ; j<i            ; j++) {

    //if same spin
    if (closed[i]%2 == closed[j]%2) {
      int I = max(closed[i]/2, closed[j]/2), 
	  J = min(closed[i]/2, closed[j]/2);

      occPairProbability.push_back( cumOccPair + I2hb.sameSpinPairExcitations(I, J) );
      occPair           .push_back( closed[i] * 2 * norbs + closed[j]);
      cumOccPair        +=  I2hb.sameSpinPairExcitations(I, J);
      //cout << I<<"  "<<J<<"  "<<cumOccPair<<"  "<<I2hb.sameSpinPairExcitations(I, J)<<endl;
    }
    else if (closed[i]%2 != closed[j]%2) {
      int I = max(closed[i]/2, closed[j]/2), 
	  J = min(closed[i]/2, closed[j]/2);

      occPairProbability.push_back( cumOccPair + I2hb.oppositeSpinPairExcitations(I, J) );
      occPair           .push_back( closed[i] * 2 * norbs + closed[j]);
      cumOccPair        +=  I2hb.oppositeSpinPairExcitations(I, J);
      //cout << I<<"  "<<J<<"  "<<cumOccPair<<"  "<<I2hb.oppositeSpinPairExcitations(I, J)<<endl;
    }
  }

  for (int nd=0; nd<ndoubles; nd++) {
    double doubleOcc       = random()*cumOccPair;
    int    doubleOccIndex  = std::lower_bound(occPairProbability.begin(), occPairProbability.end(),
					      doubleOcc) - occPairProbability.begin();

    Idouble.push_back( occPair[doubleOccIndex] / (norbs) / 2);
    Jdouble.push_back( occPair[doubleOccIndex] - Idouble[nd] * 2 * norbs);
    bool   occSameSpin     = Idouble[nd]%2 == Jdouble[nd]%2;

    int X = max(Idouble[nd], Jdouble[nd]), 
      Y = min(Idouble[nd], Jdouble[nd]);

    double pdoubletemp = occSameSpin ? 
      I2hb.sameSpinPairExcitations     (X/2, Y/2)/cumOccPair :
      I2hb.oppositeSpinPairExcitations (X/2, Y/2)/cumOccPair ;    

    double cumVirtPair = 0.;
    vector<double> virtPairProbability;
    vector<size_t> virtPair;

    virtPairProbability.reserve( open.size()*open.size());
    virtPair           .reserve( open.size()*open.size());
    
    for (int a=0; a<open.size()  ; a++) {
    for (int b=0; b<open.size()  ; b++) {
      if (open[a] %2 != Idouble[nd]%2 ||
	  open[b] %2 != Jdouble[nd]%2 )
	continue;

      int A = open[a],
	  B = open[b];
      double integral = abs(I2(A, Idouble[nd], B, Jdouble[nd]) - I2(A, Jdouble[nd], B, Idouble[nd]));
      virtPair           .push_back( A * 2 * norbs + B);
      virtPairProbability.push_back(cumVirtPair + integral);
      cumVirtPair        += integral;
    }
    }

    double doubleVirt      = random()*cumVirtPair;
    int    doubleVirtIndex = std::lower_bound(virtPairProbability.begin(), virtPairProbability.end(),
					      doubleVirt) - virtPairProbability.begin();
    
    Adouble.push_back( virtPair[doubleVirtIndex] / (norbs) / 2);
    Bdouble.push_back( virtPair[doubleVirtIndex] - Adouble[nd] * 2 * norbs);
    
    pdoubletemp *= doubleVirtIndex == 0 ? 
      abs(virtPairProbability[doubleVirtIndex]                                         )/cumVirtPair :
      abs(virtPairProbability[doubleVirtIndex] - virtPairProbability[doubleVirtIndex-1])/cumVirtPair ;
    
    pdouble.push_back(pdoubletemp);

    double integral = doubleVirtIndex == 0 ?
      abs(virtPairProbability[doubleVirtIndex]                                         ):
      abs(virtPairProbability[doubleVirtIndex] - virtPairProbability[doubleVirtIndex-1]);

  }
}

