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
#ifndef INPUT_HEADER_H
#define INPUT_HEADER_H
#include <Eigen/Dense>
#include <string>
#include <map>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include "basis.h"
#include "gaussianBasis.h"
#include "slaterBasis.h"
#include "rPseudopotential.h"
#include "iowrapper.h"

class Correlator;
class Determinant;
enum Method { sgd, amsgrad, amsgrad_sgd, ftrl, sr, linearmethod}; //varLM };
enum HAM {HUBBARD, ABINITIO};
enum BASIS {REALSPACEGTO, REALSPACESTO, ORBITALS};
enum JBASIS {NC, sNC, AB, AB2, sAB2, SS, SG, G, mG};
enum RSTEPTYPE {SIMPLE, GAUSSIAN, DMC, SPHERICAL};
enum PSEUDOQUAD {tetrahedral, octahedral, icosahedral};

/**
 * This stores all the input options
 * */

struct schedule {
private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
    ar.template register_type<gaussianBasis>();
    ar.template register_type<slaterBasis>();    
    ar & restart & fullRestart & deterministic
      & tol & correlatorFiles
      & wavefunctionType
      & numResonants
      & singleJastrow
      & readTransOrbs
      & numPermutations
      & maxIter
      & avgIter
      & printLevel
      & debug
      & decay1
      & decay2
      & alpha
      & beta
      & method
      & stochasticIter
      & sgdIter
      & momentum
      & integralSampleSize
      & seed
      & PTlambda
      & epsilon
      & screen
      & determinantFile
      & doHessian
      & hf
      & optimizeOrbs
      & optimizeCps
      & optimizeBackflow
      & optimizeJastrow
      & optimizeRBM
      & printVars
      & printGrad
      & Hamiltonian
      & ctmc
      & nwalk
      & tau
      & fn_factor
      & nGeneration
      & excitationLevel
      & numActive
      & nciAct
      & sDiagShift
      & hDiagShift
      & cgIter
      & gradTol
      & sgdStepsize
      & decay
      & stepsize 
      & stepsizes
      & walkerBasis
      & basis
      & pseudo
      & nalpha
      & nbeta
      & norbs
      & Qmax
      & QmaxEEN
      & sigma
      & realSpaceStep
      & Ncoords
      & Ncharge
      & uniqueAtoms
      & uniqueAtomsMap
      & gridGaussians
      & Nbasis
      & NSbasis
      & direct
      & rStepType
      & ifComplex
      & uagp
      & CorrSampleFrac
      & dTol
      & cgTol
      & expCorrelator
      & fourBodyJastrow
      & fourBodyJastrowBasis
      & nGrid
      & nMaxMacroIter
      & nMaxMicroIter
      & pQuad
      & nNociSlater
      & ciCeption
      & actWidth
      & overlapCutoff
      & diagMethod
      & powerShift
      & expCorrelator
      & nAttemptsEach
      & mainMemoryFac
      & spawnMemoryFac
      & shiftDamping
      & initialShift
      & minSpawn
      & minPop
      & initialPop
      & targetPop
      & numHidden
      & random
      & error
      & nVec
      & r;
  }
public:
//General options
  BASIS walkerBasis;                      //can be real space or local orbitals
  boost::shared_ptr<Basis> basis;
  boost::shared_ptr<Pseudopotential> pseudo;
  vector<Vector3d> Ncoords;
  vector<double>   Ncharge;
  vector<pair<double, Vector3d>> gridGaussians;
  vector<int> uniqueAtoms;
  vector<int> uniqueAtomsMap;
  vector<int> Nbasis;
  vector<vector<int>> NSbasis;

  bool restart;                          //option to restart calculation
  bool fullRestart;                          //option to restart calculation
  bool deterministic;                    //Performs a deterministic calculation   
  int printLevel;                        // How much stuff to print
  bool expCorrelator;                    //exponential correlator parameters, to enforce positivity
  bool debug;
  bool ifComplex;                        //breaks and restores complex conjugation symmetry 
  bool uagp;                             //brakes S^2 symmetry in uagp
  bool ciCeption;                        //true, when using ci on top of selectedCI

  int nGrid;      //the grid used for projectors in transcorrelated calcs
  int nMaxMacroIter;
  int nMaxMicroIter;
  
  RSTEPTYPE rStepType;
  PSEUDOQUAD pQuad;
  bool fourBodyJastrow;
  JBASIS fourBodyJastrowBasis;
  
//input file to define the correlator parts of the wavefunction
  int nalpha;
  int nbeta;
  int norbs;
  
  std::string wavefunctionType;
  int nNociSlater;
  std::map<int, std::string> correlatorFiles;
  std::string determinantFile;
  int Qmax;
  int QmaxEEN;
  double sigma;
  int numResonants;
  bool singleJastrow;
  bool readTransOrbs;
  int numPermutations;

//Used in the stochastic calculation of E and PT evaluation
  int stochasticIter;                    //Number of stochastic steps
  int integralSampleSize;                //This specifies the number of determinants to sample out of the o^2v^2 possible determinants after the action of V
  int seed;                              // seed for the random number generator
  double PTlambda;                       // In PT we have to apply H0- E0, here E0 = lambda x <psi0|H0|psi0> + (1 - lambda) x <psi0|H|psi0>
  double epsilon;                        // This is the usual epsilon for the heat bath truncation of integrals
  double screen;                         //This is the screening parameter, any integral below this is ignored
  bool doHessian;                        //This calcules the Hessian and overlap for the linear method
  std::string hf;
  bool optimizeOrbs;
  bool optimizeCps;
  bool optimizeBackflow;
  bool optimizeJastrow;//used in jrbm
  bool optimizeRBM;//used in jrbm
  bool printVars;
  bool printOpt;
  bool printGrad;
  HAM Hamiltonian;
  double realSpaceStep;
  
//Deprecated options for optimizers
//because now we just use the python implementation
  double tol;  
  double stepsize;
  std::vector<double> stepsizes;
  double decay1;
  double decay2;
  double alpha;
  double beta;
  double momentum;
  int maxIter;
  int maxMacroIter;
  int avgIter;                     
  int sgdIter;
  bool direct; //boolean for direct application of H, S
  Method method;
  double sDiagShift;
  double hDiagShift;
  double CorrSampleFrac;
  double decay;
  double gradTol;
  double sgdStepsize;
  double dTol;
  double cgTol;
  int cgIter;
  bool ctmc;
  bool random;
  int nVec;
  int r;
  double error;

  /*
  bool davidsonPrecondition;
  int diisSize;
  double gradientFactor;
  double mingradientFactor;
  double momentum;
  double momentumDecay;
  double decay;
  int learningEpoch;
  */

  //options for gfmc
  int nwalk;
  double tau;
  double fn_factor;
  int nGeneration;

  //options for configuration interaction
  int excitationLevel;
  int numActive; //number of active spatial orbitals, assumed to be the first in the basis
  int nciAct; //number of active spatial orbitals, assumed to be the first in the basis
  double actWidth; //used in lanczos
  double overlapCutoff; //used in SCCI
  std::string diagMethod;
  double powerShift;

  //options for FCIQMC
  int nAttemptsEach;
  double shiftDamping;
  double mainMemoryFac;
  double spawnMemoryFac;
  double initialShift;
  double minSpawn;
  double minPop;
  double initialPop;
  double targetPop;

  //options for rbm
  int numHidden;
};

/**
 * This reads the matrix of MO coefficients from 'hf.txt'
 * an alpha and a beta matrix
 * params:
 *   Matrices: matrices of the mo coefficients (nxn for rhf and uhf, 2nx2n for ghf)
 *   hf string: rhf, uhf or ghf
 */
void readHF(Eigen::MatrixXd& hforbsA, Eigen::MatrixXd& hforbsB, std::string hf);
void readHF(Eigen::MatrixXcd& hforbsA, Eigen::MatrixXcd& hforbsB, std::string hf);

/**
 * This reads the pairing matrix from 'pairMat.txt'
 * params:
 *   Matrix: matrix to be read into
 */
void readPairMat(Eigen::MatrixXd& pairMat);

/**
 * Reads the input file which by default is input.dat, but can be anything
 * else that is specified on the command line
 * 
 * params:  
 *    input:    the input file (unchanged)
 *    schd :    this is the object of class schedule that is populated by the options
 *    print:    How much to print
 */

void readMat(Eigen::MatrixXd& mat, std::string fileName);

void readMat(Eigen::MatrixXcd& mat, std::string fileName);

void readInput(const std::string inputFile, schedule& schd, bool print=true);

/**
 * We need information about the correlators because the wavefunction is
 * |Psi> = J|D>, where J is the set of jastro factors (correlators)
 * The correlator file just contains the tuple of sites that form the correlators
 * For instance, for two site correlators the file will contain lines just tell you the 
 * orbitals , e.g.
 * 0 1
 * 2 3 ... 
 * 
 * params: 
 *    input:          the input file (unchanged)
 *    correlatorSize: the size of the correlator (unchanged)
 *    correlators   : the vector of correlators,
 *                    its usually empty at input and then is filled with Correlators
 */
void readCorrelator(std::string input, int correlatorSize,
		    std::vector<Correlator>& correlators);

void readCorrelator(const std::pair<int, std::string>& p,
		    std::vector<Correlator>& correlators);


/**
 * We are just reading the set of determinants and their ci coefficients
 * for the multi-slater part of the multi-slater Jastrow wavefunction
 */
void readDeterminants(std::string input, std::vector<Determinant>& determinants,
        std::vector<double>& ciExpansion);

void readGeometry(vector<Vector3d>& Ncoords,
                  vector<double>  & Ncharge,
                  vector<int>  & uniqueAtoms,
                  vector<int>  & uniqueAtomsMap,
                  vector<int>  & Nbasis,
                  vector<vector<int>> & NSbasis,
                  gaussianBasis& gBasis);

void readGridGaussians(vector<pair<double,Vector3d>> &gridGaussians);

#endif
