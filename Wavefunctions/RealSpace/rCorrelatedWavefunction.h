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
  
  void initWalker(rWalker<Corr, Reference> &walk, Determinant &d) const 
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
    double factor1 = 1.0;
    walk.corrHelper.OverlapWithGradient(walk.d, corr, grad, factor1);
    Eigen::VectorBlock<VectorXd> gradtail = grad.tail(grad.rows() - corr.getNumVariables());
    if (schd.hf == "ghf")
      walk.OverlapWithGradientGhf(ref, gradtail);
    else
      walk.OverlapWithGradient(ref, gradtail);

  }

  void printVariables() const
  {
    corr.printVariables();
    ref.printVariables();
  }

  void updateVariables(Eigen::VectorXd &v) 
  {
    corr.updateVariables(v);
    Eigen::VectorBlock<VectorXd> vtail = v.tail(v.rows() - corr.getNumVariables());
    ref.updateVariables(vtail);
  }

  void getVariables(Eigen::VectorXd &v) const
  {
    if (v.rows() != getNumVariables())
      v = VectorXd::Zero(getNumVariables());

    corr.getVariables(v);
    Eigen::VectorBlock<VectorXd> vtail = v.tail(v.rows() - corr.getNumVariables());
    ref.getVariables(vtail);
  }


  long getNumJastrowVariables() const
  {
    return corr.getNumVariables();
  }
  //factor = <psi|w> * prefactor;

  long getNumVariables() const
  {
    int norbs = Determinant::norbs;
    long numVars = 0;
    numVars += getNumJastrowVariables();
    numVars += ref.getNumVariables();

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
    return walk.corrHelper.OverlapRatio(i, coord, corr, walk.d)
        * walk.refHelper .getDetFactor(i, coord, walk.d, ref);
  }

  //<psi_t| (H-E0) |D>
  //This function is used with orbital basis
  double rHam(const rWalker<Corr, Reference> &walk) const
  {
    int norbs = Determinant::norbs;

    double potentialij = 0.0, potentiali=0;

    //get potential
    for (int i=0; i<walk.d.nelec; i++)
      for (int j=i+1; j<walk.d.nelec; j++) {
        potentialij += 1./walk.corrHelper.Rij(i,j);
      }

    for (int i=0; i<walk.d.nelec; i++) {
      for (int j=0; j<schd.Ncoords.size(); j++) {
        potentiali -= schd.Ncharge[j]/walk.corrHelper.RiN(i,j);
      }
    }


    if (schd.hf == "ghf" ) {
      double kinetic = 0.0;

      {
        MatrixXd Bij = walk.refHelper.Laplacian[0]; //i = nelec , j = norbs
        
        for (int i=0; i<walk.d.nalpha+walk.d.nbeta; i++) 
          Bij.row(i) += 2.*walk.corrHelper.GradRatio.row(i) * walk.refHelper.Gradient[i];
        
        for (int i=0; i<walk.d.nalpha+walk.d.nbeta; i++) {
          kinetic += Bij.row(i).dot(walk.refHelper.thetaInv[0].col(i));
          kinetic += walk.corrHelper.LaplaceRatio[i];
        }
        //cout << " k "<<walk.corrHelper.LaplaceRatio[0]<<endl;
      }
      //return potentialij;
      //cout << walk.corrHelper.Rij(1,0)<<"  "<<walk.corrHelper.Rij(2,0)<<"  "<<walk.corrHelper.Rij(3,0)<<"  "<<walk.corrHelper.Rij(2,1)<<"  "<<walk.corrHelper.Rij(3,1)<<"  "<<walk.corrHelper.Rij(3,2)<<endl;
      //cout << walk.corrHelper.RiN(0,0)<<"  " << walk.corrHelper.RiN(1,0)<<"  " << walk.corrHelper.RiN(2,0)<<"  " << walk.corrHelper.RiN(3,0)<<"  " <<endl;
      //cout << -0.5*(kinetic) <<"  "<< potentialij<<"  "<<potentiali<<"  "<< -0.5*(kinetic) + potentialij+potentiali<<endl;
      return -0.5*(kinetic) + potentialij+potentiali;

    }
    else {
      
      double kinetica = 0.0;
      //Alpha
      {
        MatrixXd Bij = walk.refHelper.Laplacian[0]; //i = nelec , j = norbs
        //cout << " k "<<walk.refHelper.thetaInv[0](0,0)*Bij(0,0)<<endl;;
        
        for (int i=0; i<walk.d.nalpha; i++) 
          Bij.row(i) += 2.*walk.corrHelper.GradRatio.row(i) * walk.refHelper.Gradient[i];
        
        //cout << " k "<<walk.corrHelper.GradRatio.row(0) <<"  "<< walk.refHelper.Gradient[0]<<endl;
        for (int i=0; i<walk.d.nalpha; i++) {
          kinetica += Bij.row(i).dot(walk.refHelper.thetaInv[0].col(i));
          kinetica += walk.corrHelper.LaplaceRatio[i];
        }
        //cout << " k "<<walk.corrHelper.LaplaceRatio[0]<<endl;
      }
      
      double kineticb = 0.0;
      //Beta
      if (walk.d.nbeta != 0)
      {
        MatrixXd Bij = walk.refHelper.Laplacian[1]; //i = nelec , j = norbs
        int nalpha = walk.d.nalpha;
        //cout << " k "<<walk.refHelper.thetaInv[1](0,0)*Bij(0,0)<<endl;;
        
        for (int i=0; i<walk.d.nbeta; i++) 
          Bij.row(i) += 2*walk.corrHelper.GradRatio.row(i+nalpha) * walk.refHelper.Gradient[i+nalpha];
        
        for (int i=0; i<walk.d.nbeta; i++) {
          kineticb += Bij.row(i).dot(walk.refHelper.thetaInv[1].col(i));
          kineticb += walk.corrHelper.LaplaceRatio[i+nalpha];
        }
        //cout << " k "<<walk.corrHelper.LaplaceRatio[1]<<endl;
      }
      return -0.5*(kinetica+kineticb) + potentialij+potentiali;
    }

    //return potentialij;
  }
  
  
};


#endif
