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
#ifndef rCorrelatedWavefunction_HEADER_H
#define rCorrelatedWavefunction_HEADER_H
#include <vector>
#include <set>
#include <boost/serialization/serialization.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include "rWalker.h"
#include "rWalkerHelper.h"

class rDeterminant;


/**
 * This is the wavefunction, it is a product of the CPS and a linear combination of
 * slater determinants
 */
template<typename Corr, typename Reference>  //Corr = CPS/JAstrow or Reference = RHF/U
struct rCorrelatedWavefunction {
 private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & corr
        & ref;
  }

 public:
  using CorrType = Corr;
  using ReferenceType = Reference;
  
  Corr corr; //The jastrow factors
  Reference ref; //reference

  rCorrelatedWavefunction() {};
  Reference& getRef() { return ref; }
  Corr& getCorr() { return corr; }

  void initWalker(rWalker<Corr, Reference> &walk) const 
  {
    walk = rWalker<Corr, Reference>(corr, ref);
  }
  
  void initWalker(rWalker<Corr, Reference> &walk, rDeterminant &d) const 
  {
    walk = rWalker<Corr, Reference>(corr, ref, d);
  }
  
  double Overlap(const rWalker<Corr, Reference> &walk) const 
  {
    return exp(walk.corrHelper.exponential) * walk.getDetOverlap(ref);
  }


  /**
   * This basically calls the overlapwithgradient(determinant, factor, grad)
   */
  void OverlapWithGradient(rWalker<Corr, Reference> &walk,
                           double &factor,
                           Eigen::VectorXd &grad) const
  {
    walk.OverlapWithGradient(ref, corr, grad);
  }

  void printVariables() const
  {
    corr.printVariables();
    ref.printVariables();
  }

  void updateVariables(Eigen::VectorXd &v) 
  {
    corr.updateVariables(v);
    Eigen::VectorBlock<VectorXd> vtail = v.tail(getNumRefVariables());
    ref.updateVariables(vtail);

    if (schd.walkerBasis == REALSPACESTO && schd.enforceENCusp)
      enforceCusp();
  }

  void updateOptVariables(Eigen::VectorXd &v) 
  {
    if (schd.optimizeCps) {
      corr.updateVariables(v);
    }
    if (schd.optimizeOrbs) {
      Eigen::VectorBlock<VectorXd> vtail = v.tail(getNumRefVariables());
      ref.updateVariables(vtail);
    }

    if (schd.walkerBasis == REALSPACESTO && schd.enforceENCusp)
      enforceCusp();
  }


  void getVariables(Eigen::VectorXd &v) const
  {
    v.setZero(getNumVariables());

    corr.getVariables(v);
    Eigen::VectorBlock<VectorXd> vtail = v.tail(getNumRefVariables());
    ref.getVariables(vtail);
  }

  void getOptVariables(Eigen::VectorXd &v) const
  {
    v.setZero(getNumOptVariables());
    if (schd.optimizeCps) {
      corr.getVariables(v);
    }
    if (schd.optimizeOrbs) {
      Eigen::VectorBlock<VectorXd> vtail = v.tail(getNumRefVariables());
      ref.getVariables(vtail);
    }
  }

  long getNumJastrowVariables() const
  {
    return corr.getNumVariables();
  }

  long getNumRefVariables() const
  {
    return ref.getNumVariables();
  }

  long getNumVariables() const
  {
    long numVars = 0;
    numVars += getNumJastrowVariables();
    numVars += getNumRefVariables();
    return numVars;
  }

  long getNumOptVariables() const
  {
    long numVars = 0;
    if (schd.optimizeCps) numVars += getNumJastrowVariables();
    if (schd.optimizeOrbs) numVars += getNumRefVariables();
    return numVars;
  }

  string getfileName() const {
    return ref.getfileName() + corr.getfileName();
  }
  
  void writeWave() const
  {
    if (commrank == 0)
    {
      char file[5000];
      //sprintf (file, "wave.bkp" , schd.prefix[0].c_str() );
      sprintf(file, (getfileName()+".bkp").c_str() );
      std::ofstream outfs(file);
      boost::archive::text_oarchive save(outfs);
      save << *this;
      outfs.close();
    }
  }

  void readWave()
  {
    if (commrank == 0)
    {
      char file[5000];
      //sprintf (file, "wave.bkp" , schd.prefix[0].c_str() );
      sprintf(file, (getfileName()+".bkp").c_str() );
      std::ifstream infs(file);
      boost::archive::text_iarchive load(infs);
      load >> *this;
      infs.close();
    }
#ifndef SERIAL
    boost::mpi::communicator world;
    boost::mpi::broadcast(world, *this, 0);
#endif
  }


  double getOverlapFactor(int i, Vector3d& coord, rWalker<Corr, Reference>& walk) const 
  {
    return walk.corrHelper.OverlapRatio(i, coord, corr, walk.d) * walk.refHelper.getDetFactor(i, coord, walk.d, ref);
  }

  //<psi_t| (H-E0) |D>
  //This function is used with orbital basis
  double rHam(rWalker<Corr, Reference> &walk) const
  {
    cout << "Should not be here. There is a specialized rHam for various cases "<<endl;
    exit(0);
    return 0;
  }

  double rHam(rWalker<Corr, Reference>& walk, double& T, double& Vij, double& ViI, double& Vpp, double& VIJ, std::vector<std::vector<double>> &Viq, std::vector<std::vector<Vector3d>> &Riq) const
  {
    cout << "Should not be here. There is a specialized rHam for various cases "<<endl;
    exit(0);
    return 0;
  }
  
  double HamOverlap(rWalker<rJastrow, rSlater>& walk,
                  Eigen::VectorXd &gradRatio,
                  Eigen::VectorXd &hamRatio) const
  {
    cout << "Should not be here. There is a specialized rHam for various cases "<<endl;
    exit(0);
    return 0;
  }

  double HamOverlap(rWalker<rJastrow, rBFSlater>& walk,
                  Eigen::VectorXd &gradRatio,
                  Eigen::VectorXd &hamRatio) const
  {
    cout << "Should not be here. There is a specialized rHam for various cases "<<endl;
    exit(0);
    return 0;
  }
  
  void enforceCusp()
  {
    cout << "Should not be here. There is a specialized enforceCusp for various cases "<<endl;
    exit(0);
    return 0;
  }

  double getDMCMove(Vector3d& coord, int elecI, double stepsize,
                    rWalker<rJastrow, rSlater>& walk);
};


template<>
double rCorrelatedWavefunction<rJastrow, rSlater>::rHam(rWalker<rJastrow, rSlater>& walk, double& T, double& Vij, double& ViI, double& Vpp, double& VIJ, std::vector<std::vector<double>> &Viq, std::vector<std::vector<Vector3d>> &Riq) const;

template<>
double rCorrelatedWavefunction<rJastrow, rSlater>::rHam(rWalker<rJastrow, rSlater>& walk) const;

template<>
double rCorrelatedWavefunction<rJastrow, rSlater>::HamOverlap(rWalker<rJastrow, rSlater>& walk,
                                                              Eigen::VectorXd& gradRatio,
                                                              Eigen::VectorXd& hamRatio) const;

template<>
void rCorrelatedWavefunction<rJastrow, rSlater>::enforceCusp();


template<>
double rCorrelatedWavefunction<rJastrow, rBFSlater>::rHam(rWalker<rJastrow, rBFSlater>& walk) const;

template<>
double rCorrelatedWavefunction<rJastrow, rBFSlater>::HamOverlap(rWalker<rJastrow, rBFSlater>& walk,
                                                              Eigen::VectorXd& gradRatio,
                                                              Eigen::VectorXd& hamRatio) const;

template<>
void rCorrelatedWavefunction<rJastrow, rBFSlater>::enforceCusp();

#endif
