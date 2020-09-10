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
#include "rSlater.h"
#include "rBFSlater.h"
#include <random>

template<typename jastrow, typename slater> class rCorrelatedWavefunction;
using namespace Eigen;

template<typename T1, typename T2>
struct rWalker {
};

using uniformRandom = std::function<double ()>;
using normalRandom = std::function<double ()>;

template<>
struct rWalker<rJastrow, rSlater> {

  rDeterminant d;
  MatrixXd RNM;
  MatrixXd Rij;         //the inter-electron distances
  MatrixXd RiN;         //electron-nucleus distances  
  MatrixXcd Bnl;        //Bnl(elec, mo) = Vnl * DetMatrix     
  MatrixXd AOBnl;        //AOBnl(elec, ao) = \partial_{mocoeff} Vnl * DetMatrix    
  std::vector<Vector3d> Q;  //qudrature points
  rWalkerHelper<rJastrow> corrHelper;
  rWalkerHelper<rSlater> refHelper;
  uniformRandom uR;
  std::normal_distribution<double> nR;
  
  rWalker();
  
  rWalker(const rJastrow &corr, const rSlater &ref) ;

  rWalker(const rJastrow &corr, const rSlater &ref, const rDeterminant &pd);

  void initR();

  void initHelpers(const rJastrow &corr, const rSlater &ref);

  void initBnl(const rJastrow &corr, const rSlater &ref, double &local_potential);

  void updateBnl(int nelec, const rJastrow &corr, const rSlater &ref);
  
  rDeterminant& getDet();

  void readBestDeterminant(rDeterminant& d) const ;

  double getDetOverlap(const rSlater &ref) const;

  /**
   * makes det based on mo coeffs 
   */
  void guessBestDeterminant(rDeterminant& d, const Eigen::MatrixXcd& HforbsA, const Eigen::MatrixXcd& HforbsB) const ;

  void initDet(const MatrixXcd& HforbsA, const MatrixXcd& HforbsB) ;


  void updateWalker(int elec, Vector3d& coord, const rSlater& ref, const rJastrow& corr);

  void OverlapWithGradient(const rSlater &ref, const rJastrow& cps, VectorXd &grad) ;

  void HamOverlap(const rSlater &ref, const rJastrow& cps, VectorXd &grad) ;

  void getStep(Vector3d& coord, int elecI, double stepsize,
               const rSlater& ref, const rJastrow& corr, double& ovlpRatio,
               double& proposalProb) ;
               
  void getSimpleStep(Vector3d& coord,  double stepsize, double& ovlpRatio, double& proposalProb);

  void getSphericalStep(Vector3d& coord, int elecToMove, double stepsize, const rSlater& ref,
                        double& ovlpRatio, double& proposalProb);
  void getDMCStep(Vector3d& coord, int elecI, double stepsize,
                 const rSlater& ref, const rJastrow& corr, double& ovlpRatio,
                 double& proposalProb) ;
  void getGaussianStep(Vector3d& coord, int elecToMove, double stepsize,
                       double& ovlpRatio, double& proposalProb);

  void doDMCMove(Vector3d& coord, int elecI, double stepsize, const rSlater& ref, const rJastrow& corr, double& ovlpRatio, double& proposalProb);

  void getGradient(int elecI, Vector3d& grad) ;
  double getGradientAfterSingleElectronMove(int elecI, Vector3d& move, Vector3d& grad,
                                            const rSlater& ref) ;
  double getGradientAfterSingleElectronMovetemp(int elecI, Vector3d& move, Vector3d& grad,
                                            const rSlater& ref) ;
  
  friend ostream& operator<<(ostream& os, const rWalker<rJastrow, rSlater>& w) {
    os << w.d << endl << endl;
    os << "dets\n" << w.refHelper.thetaDet[0][0] << "  " << w.refHelper.thetaDet[0][1] << endl << endl;
    os << "alphaInv\n" << w.refHelper.thetaInv[0] << endl << endl;
    os << "betaInv\n" << w.refHelper.thetaInv[1] << endl << endl;
    return os;
  }
  
};

template<>
struct rWalker<rJastrow, rBFSlater> {

  rDeterminant d;
  MatrixXd RNM;
  MatrixXd Rij;         //the inter-electron distances
  MatrixXd RiN;         //electron-nucleus distances  
  rWalkerHelper<rJastrow> corrHelper;
  rWalkerHelper<rBFSlater> refHelper;
  uniformRandom uR;
  std::normal_distribution<double> nR;
  
  rWalker();
  
  rWalker(const rJastrow &corr, const rBFSlater &ref) ;

  rWalker(const rJastrow &corr, const rBFSlater &ref, const rDeterminant &pd);

  void initR();
  void initHelpers(const rJastrow &corr, const rBFSlater &ref);
  
  rDeterminant& getDet();
  void readBestDeterminant(rDeterminant& d) const ;


  double getDetOverlap(const rBFSlater &ref) const;

  /**
   * makes det based on mo coeffs 
   */
  void guessBestDeterminant(rDeterminant& d, const Eigen::MatrixXcd& HforbsA, const Eigen::MatrixXcd& HforbsB) const ;

  void initDet(const MatrixXcd& HforbsA, const MatrixXcd& HforbsB) ;


  void updateWalker(int elec, Vector3d& coord, const rBFSlater& ref, const rJastrow& corr);

  void OverlapWithGradient(const rBFSlater &ref, const rJastrow& cps, VectorXd &grad) ;

  void HamOverlap(const rBFSlater &ref, const rJastrow& cps, VectorXd &grad) ;

  void getStep(Vector3d& coord, int elecI, double stepsize,
               const rBFSlater& ref, const rJastrow& corr, double& ovlpRatio,
               double& proposalProb) ;
               
  void getSimpleStep(Vector3d& coord,  double stepsize, double& ovlpRatio, double& proposalProb);

  void getSphericalStep(Vector3d& coord, int elecToMove, double stepsize, const rBFSlater& ref,
                        double& ovlpRatio, double& proposalProb);
  void doDMCMove(Vector3d& coord, int elecI, double stepsize,
                 const rBFSlater& ref, const rJastrow& corr, double& ovlpRatio,
                 double& proposalProb) ;
  void getGaussianStep(Vector3d& coord, int elecToMove, double stepsize,
                       double& ovlpRatio, double& proposalProb);

  void getGradient(int elecI, Vector3d& grad) ;
  double getGradientAfterSingleElectronMove(int elecI, Vector3d& move, Vector3d& grad,
                                            const rBFSlater& ref) ;
  
  friend ostream& operator<<(ostream& os, const rWalker<rJastrow, rBFSlater>& w) {
    os << "bare det\n" << w.d << endl << endl;
    os << "quasiparticle det\n" << w.refHelper.dp << endl << endl;
    os << "DetMatrix\n" << w.refHelper.DetMatrix << endl << endl;
    os << "det overlap\n" << w.refHelper.thetaDet << endl << endl;
    os << "hValues\n" << w.refHelper.hValues << endl << endl;
    os << "etaValues\n" << w.refHelper.etaValues << endl << endl;
    os << "hGradient0\n" << w.refHelper.hGradient[0] << endl << endl;
    os << "hGradient1\n" << w.refHelper.hGradient[1] << endl << endl;
    os << "hGradient2\n" << w.refHelper.hGradient[2] << endl << endl;
    for (int i = 0; i < 9; i++) {
      os << "rpGradient " << i << endl << w.refHelper.rpGradient[i] << endl << endl;
    }
    for (int i = 0; i < 9; i++) {
      os << "rpSecondDerivatives " << i << endl << w.refHelper.rpSecondDerivatives[i] << endl << endl;
    }
    for (int i = 0; i < 3; i++) {
      os << "MOGradient " << i << endl << w.refHelper.MOGradient[i] << endl << endl;
    }
    os << "thetaInv\n" << w.refHelper.thetaInv << endl << endl;
    for (int i = 0; i < 3; i++) {
      os << "rTable " << i << endl << w.refHelper.rTable[i] << endl << endl;
    }
    for (int i = 0; i < 6; i++) {
      os << "MOSecondDerivatives " << i << endl << w.refHelper.MOSecondDerivatives[i] << endl << endl;
    }
    for (int i = 0; i < 3; i++) {
      os << "slaterGradientRatio " << i << endl << w.refHelper.slaterGradientRatio[i] << endl << endl;
    }
    os << "slaterLaplacianRatio\n" << w.refHelper.slaterLaplacianRatio << endl << endl;
    return os;
  }
  
};


struct SphericalSteps {
  static double distance(Vector3d& r1, Vector3d& r2);

  static int findTheNearestNucleus(Vector3d& ri, double& riN);

  static void RotateTowards(Vector3d& R, Vector3d& ri, Vector3d& rf);

  static void initializeU(Vector3d& grad, double& eta,
                          double& a, Vector3d& di, int N);

  static double RejectionSampleR(double& eta, double& a, double& dR, double& RiN,
                                 uniformRandom& uR) ;

  static double SamplePhi(double& rif, double& thetaf, Vector3d& ri, Vector3d& grad,
                          int N, uniformRandom& uR);
  static double G(double& eta, double& a, double x, double y) ;
  static double volume(double& a, double& eta, double& Ri, double& dR, double& thetam);
  
};

#endif
