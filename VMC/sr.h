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
#include "input.h"
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
      VectorXd Vectors_bar = VectorXd::Zero(dim);
      Smatrix = MatrixXd::Zero(dim, dim);
      for (int i = 0; i < Vectors.size(); i++)
      {
        Smatrix += T[i] * Vectors[i] * Vectors[i].adjoint();
        Vectors_bar += T[i] * Vectors[i];
        Tau += T[i];
      }

#ifndef SERIAL
      MPI_Allreduce(MPI_IN_PLACE, &(Tau), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(Vectors_bar(0)), Vectors_bar.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(Smatrix(0,0)), Smatrix.rows() * Smatrix.cols(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
      Vectors_bar /= Tau;
      Smatrix /= Tau; 
      Smatrix -= Vectors_bar * Vectors_bar.adjoint();
    }
      
    
  void multiply(VectorXd &x, VectorXd& Ax)
  {
    double Tau = 0.0;
    int dim = x.rows();
    VectorXd Vectors_bar = VectorXd::Zero(dim);
    Ax.setZero(dim);
    for (int i = 0; i < Vectors.size(); i++)
    {
      double factor = Vectors[i].adjoint() * x;
      Ax += T[i] * Vectors[i] * factor;
      Vectors_bar += T[i] * Vectors[i];
      Tau += T[i];
    }
#ifndef SERIAL
      MPI_Allreduce(MPI_IN_PLACE, &(Vectors_bar(0)), Vectors_bar.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(Ax(0)), Ax.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(Tau), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
    Vectors_bar /= Tau;
    Ax /= Tau;

    Ax += diagshift * x;
    double factor = Vectors_bar.adjoint() * x;
    Ax -= Vectors_bar * factor;
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
        ar & iter;
    }

  public:
    int maxIter;
    int iter;

    SR(int _maxIter=1000) : maxIter(_maxIter)
    {
        iter = 0;
    }

    void write(VectorXd& vars)
    {
        if (commrank == 0)
        {
	    char file[50];
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
       DirectMetric S(schd.sDiagShift);
       double E0 = 0.0, stddev = 0.0, rt = 0.0;
       if (!schd.direct) { S.Smatrix = MatrixXd::Zero(numVars, numVars); }

       getMetric(vars, grad, S, E0, stddev, rt);
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

       VectorXd x = VectorXd::Zero(numVars);
       if (schd.direct)
       {
         ConjGrad(S, grad, schd.cgIter, x);
       }
       else
       {
         if (commrank == 0)
         {
           S.Smatrix += schd.sDiagShift * MatrixXd::Identity(numVars, numVars);
           MatrixXd SInv = MatrixXd::Zero(numVars, numVars);
           PInv(S.Smatrix, SInv);
           x = SInv * grad;
         }
       }
       
       if (commrank == 0)
       {
         //update vars
         vars -= schd.stepsize * x;
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
