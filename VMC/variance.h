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
#ifndef OPTIMIZERVARIANCE_HEADER_H
#define OPTIMIZERVARIANCE_HEADER_H
#include <Eigen/Dense>
#include <boost/serialization/serialization.hpp>
#include "iowrapper.h"
#include "global.h"
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#include <fstream>
#include "sr.h"

#ifndef SERIAL
#include "mpi.h"
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif

using namespace Eigen;
using namespace boost;
using namespace std;

class DirectVarianceHessian
{
    public:
    vector<double> T;
    vector<VectorXd> grad_Eloc;
    VectorXd grad_Energy;
    double diagshift;
    MatrixXd Hessian;
    DirectVarianceHessian(double _diagshift) : diagshift(_diagshift) {}

    void BuildMatrix()
    {
      double Tau = 0.0;
      int dim = grad_Eloc[0].rows();
      Hessian = MatrixXd::Zero(dim, dim);
      VectorXd grad_Eloc_bar = VectorXd::Zero(dim);
      for (int i = 0; i < T.size(); i++)
      {
        Tau += T[i];
        Hessian += T[i] * (grad_Eloc[i] * grad_Eloc[i].adjoint() - Hessian) / Tau;
        grad_Eloc_bar += T[i] * (grad_Eloc[i] - grad_Eloc_bar) / Tau;
      }
#ifndef SERIAL
      MPI_Allreduce(MPI_IN_PLACE, &(Hessian(0,0)), Hessian.rows() * Hessian.cols(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, (grad_Eloc_bar.data()), grad_Eloc_bar.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
      Hessian /= commsize;
      grad_Eloc_bar /= commsize;
      Hessian = Hessian - grad_Energy * grad_Eloc_bar.adjoint() - grad_Eloc_bar * grad_Energy.adjoint() + grad_Energy * grad_Energy.adjoint();
    }
  
  void multiply(const VectorXd &x, VectorXd& Hx) const
  {
    double Tau = 0.0;
    int dim = x.rows();
    Hx.setZero(dim);
    VectorXd grad_Eloc_bar = VectorXd::Zero(dim);
    for (int i = 0; i < T.size(); i++)
    {
      Tau += T[i];
      grad_Eloc_bar += T[i] * (grad_Eloc[i] - grad_Eloc_bar) / Tau;
      Hx += T[i] * (grad_Eloc[i] * grad_Eloc[i].dot(x) - Hx) / Tau;
    }
#ifndef SERIAL
      MPI_Allreduce(MPI_IN_PLACE, &(Hx(0)), Hx.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(grad_Eloc_bar(0)), grad_Eloc_bar.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
      Hx /= commsize;
      grad_Eloc_bar /= commsize;
      Hx -= grad_Energy * grad_Eloc_bar.dot(x);
      Hx -= grad_Eloc_bar * grad_Energy.dot(x);
      Hx += grad_Energy * grad_Energy.dot(x);
      Hx += diagshift * x;
  } 
};

void VarConjGrad(const DirectVarianceHessian &A, const VectorXd &b, int n, VectorXd &x)
{
  double tol = 1.e-10;

  VectorXd Ap = VectorXd::Zero(x.rows());
  A.multiply(x, Ap);
  VectorXd r = b - Ap;
  VectorXd p = r;
  
  double rsold = r.adjoint() * r;
  if (fabs(rsold) < tol) return;
  
  for (int i = 0; i < n; i++)
  {
    A.multiply(p, Ap);
    double pAp = p.adjoint() * Ap;
    double alpha = rsold / pAp;

    x = x + alpha * p;
    r = r - alpha * Ap;
    
    double rsnew = r.adjoint() * r;
    double beta = rsnew / rsold;

    rsold = rsnew;
    p = r + beta*p;
    if (fabs(rsold) < tol) return;
  }
}

//void PInv(MatrixXd &A, MatrixXd &Ainv);

class Variance
{
  private:
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
        ar & stepsize & iter;
    }

  public:
    double stepsize;
    int maxIter;
    int iter;

    Variance(double pstepsize=0.01, int pmaxIter=1000) : stepsize(pstepsize), maxIter(pmaxIter)
    {
        iter = 0;
    }

    void write(VectorXd& vars)
    {
        if (commrank == 0)
        {
	    char file[5000];
            sprintf(file, "variance.bkp");
            std::ofstream ofs(file, std::ios::binary);
            boost::archive::binary_oarchive save(ofs);
            save << *this;
            save << vars;
            ofs.close();
        }
    }

    void read(VectorXd& vars)
    {
        if (commrank == 0)
        {
	    char file[50];
            sprintf(file, "variance.bkp");
            std::ifstream ifs(file, std::ios::binary);
	    boost::archive::binary_iarchive load(ifs);
            load >> *this;
            load >> vars;
            ifs.close();
        }
    }

   template<typename Function>
   void optimize(VectorXd &vars, Function &getVariance, bool restart)
   {
     if (restart)
     {
       if (commrank == 0) 
         read(vars);
#ifndef SERIAL
	    boost::mpi::communicator world;
	    boost::mpi::broadcast(world, *this, 0);
	    boost::mpi::broadcast(world, vars, 0);
#endif
     }
     int numVars = vars.rows();
     VectorXd grad, x;
     DirectVarianceHessian H(schd.hDiagShift);
     double E0, stddev, rt, variance;
     rt = 0.0;
     while (iter < maxIter)
     {
       E0 = 0.0;
       variance = 0.0;
       stddev = 0.0;
       grad.setZero(numVars);
       x.setZero(numVars);
       if (iter < schd.sgdIter)
         rt = 0.0;
       getVariance(vars, grad, H, variance, E0, stddev, rt);
       auto VMC_time = (getTime() - startofCalc);
       write(vars);

       if (!schd.direct)
       {
         H.BuildMatrix();
         H.Hessian += MatrixXd::Identity(numVars, numVars);
       }

       if (iter < schd.sgdIter)
       {
         vars += -0.1 * grad;
       }
       else 
       {
         if (schd.direct)
         {
           x << -(0.01 * grad);
           VarConjGrad(H, -grad, schd.cgIter, x);
         }
         else
         {
           if (commrank == 0)
           {
             MatrixXd HInv = MatrixXd::Zero(numVars, numVars);
             PInv(H.Hessian, HInv);
             x = HInv * grad;
           }
         }
         
         if (commrank == 0)
         {
           //update vars
           for (int i = 0; i < x.rows(); i++)
           {
             vars(i) += schd.stepsize * x(i);
           }
         }
       }
#ifndef SERIAL
       MPI_Bcast(&(vars[0]), vars.rows(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
       if (commrank == 0)
         std::cout << format("%5i %14.8f %14.8f (%8.2e) %14.8f %8.1f %8.2f %8.2f\n") % iter % variance % E0 % stddev % (grad.norm()) % (rt) % (VMC_time) % ((getTime() - startofCalc));
       iter++;
     }
   }
};   

#endif
