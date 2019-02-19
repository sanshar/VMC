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
#ifndef OPTIMIZERSR_HEADER_H
#define OPTIMIZERSR_HEADER_H
#include <Eigen/Dense>
#include <boost/serialization/serialization.hpp>
#include "iowrapper.h"
#include "global.h"
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#include <fstream>

#ifndef SERIAL
#include "mpi.h"
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif

using namespace Eigen;
using namespace boost;
using namespace std;

class DirectMetric
{
    public:
    vector<double> T;
    vector<VectorXd> Vectors;
    double diagshift;
    MatrixXd Smatrix;
    DirectMetric(double _diagshift) : diagshift(_diagshift) {}

    void BuildMetric()
    {
      double Tau = 0.0;
      int dim = Vectors[0].rows();
      Smatrix = MatrixXd::Zero(dim, dim);
      for (int i = 0; i < Vectors.size(); i++)
      {
        Smatrix += T[i] * Vectors[i] * Vectors[i].adjoint();
        Tau += T[i];
      }

#ifndef SERIAL
      MPI_Allreduce(MPI_IN_PLACE, &(Tau), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(Smatrix(0,0)), Smatrix.rows() * Smatrix.cols(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
      Smatrix /= Tau; 
    }
      
    
  void multiply(VectorXd &x, VectorXd& Ax)
  {
    double Tau = 0.0;
    int dim = x.rows();
    if (Ax.rows() != x.rows())
      Ax = VectorXd::Zero(dim);
    else
      Ax.setZero();
    for (int i = 0; i < Vectors.size(); i++)
    {
      double factor = Vectors[i].adjoint() * x;
      Ax += T[i] * Vectors[i] * factor;
      Tau += T[i];
    }
#ifndef SERIAL
      MPI_Allreduce(MPI_IN_PLACE, &(Ax(0)), Ax.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(Tau), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
      Ax /= Tau;
      Ax += diagshift * x;
    } 
};

void ConjGrad(DirectMetric &A, VectorXd &b, int n, VectorXd &x);

void PInv(MatrixXd &A, MatrixXd &Ainv);

class SR
{
  private:
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
        ar & stepsize
            & iter;
    }

  public:
    double stepsize;
    int maxIter;
    int iter;

    SR(double pstepsize=0.01, int pmaxIter=1000) : stepsize(pstepsize), maxIter(pmaxIter)
    {
        iter = 0;
    }

    void write(VectorXd& vars)
    {
        if (commrank == 0)
        {
	    char file[5000];
            sprintf(file, "sr.bkp");
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
            sprintf(file, "sr.bkp");
            std::ifstream ifs(file, std::ios::binary);
	    boost::archive::binary_iarchive load(ifs);
            load >> *this;
            load >> vars;
            ifs.close();
        }
    }

   template<typename Function>
   void optimize(VectorXd &vars, Function &getMetric, bool restart)
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
     while (iter < maxIter)
     {
       VectorXd grad = VectorXd::Zero(numVars);
       VectorXd x, H;
       DirectMetric S(schd.sDiagShift);
       double E0 = 0.0, stddev = 0.0, rt = 0.0;
       x.setZero(numVars + 1);
       if (!schd.direct)
       {
         S.Smatrix = schd.sDiagShift * MatrixXd::Identity(numVars + 1, numVars + 1);
       }

       getMetric(vars, grad, H, S, E0, stddev, rt);
       write(vars);
       auto VMC_time = (getTime() - startofCalc);

       /*
         FOR DEBUGGING PURPOSE
       */
       /*
       S.BuildMetric();
       for(int i=0; i<vars.rows(); i++)
         S.Smatrix(i,i) += 1.e-4;
       MatrixXd s_inv = MatrixXd::Zero(vars.rows() + 1, vars.rows() + 1);
       PIn(S.Smatrix,s_inv);
       x = s_inv * s.H;
       */
       
       //cout << s.H << endl<<endl;
       //cout << x <<endl<<endl<<endl;;

       //xguess << 1.0, vars;

       if (schd.direct)
       {
         x[0] = 1.0;
         ConjGrad(S, H, schd.cgIter, x);
       }
       else
       {
         if (commrank == 0)
         {
           MatrixXd SInv = MatrixXd::Zero(numVars + 1, numVars + 1);
           PInv(S.Smatrix, SInv);
           x = SInv * H;
         }
       }
       
       if (commrank == 0)
       {
         //update vars
         for (int i = 0; i < vars.rows(); i++)
         {
           vars(i) += (x(i+1) / x(0));
         }
       }
#ifndef SERIAL
       MPI_Bcast(&(vars[0]), vars.rows(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
       if (commrank == 0)
         std::cout << format("%5i %14.8f (%8.2e) %14.8f %8.1f %8.2f %8.2f\n") % iter % E0 % stddev % (grad.norm()) % (rt) % (VMC_time) % ((getTime() - startofCalc));
       iter++;
     }
   }
};   

#endif
