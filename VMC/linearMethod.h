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
#ifndef OPTIMIZERLM_HEADER_H
#define OPTIMIZERLM_HEADER_H
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


class LM
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

    LM(double pstepsize=0.001, int pmaxIter=1000) : stepsize(pstepsize), maxIter(pmaxIter)
    {
        iter = 0;
    }

    void write(VectorXd& vars)
    {
        if (commrank == 0)
        {
	    char file[5000];
            sprintf(file, "lm.bkp");
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
	    char file[5000];
            sprintf(file, "lm.bkp");
            std::ifstream ifs(file, std::ios::binary);
	    boost::archive::binary_iarchive load(ifs);
            load >> *this;
            load >> vars;
            ifs.close();
        }
    }

   template<typename Function>
   void optimize(VectorXd &vars, Function& getHessian, bool restart)
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
     MatrixXd Smatrix, Hessian;
     //DirectMetric s(schd.sDiagShift);

     double E0, stddev, rt;
     while (iter < maxIter)
     {
       E0 = 0.0;
       stddev = 0.0;
       rt = 0.0;
       grad.setZero(numVars);
       x.setZero(numVars + 1);

       getHessian(vars, grad, Hessian, Smatrix, E0, stddev, rt);
       write(vars);

       for (int i=0; i<numVars+1; i++)
         Hessian(i,i) += schd.sDiagShift;

       MatrixXd Uo = MatrixXd::Zero(numVars+1, numVars+1);       
       Uo(0,0) = 1.0;
       for (int i=0; i<numVars; i++) {
         Uo(0, i+1) = -Smatrix(0, i+1);
         Uo(i+1, i+1) = 1.0;
       }
       
       Smatrix = Uo.transpose()*(Smatrix * Uo);
       Hessian = Uo.transpose()*(Hessian * Uo);
       SelfAdjointEigenSolver<MatrixXd> oes(Smatrix);

       int index = 0;
       Uo.setZero();
       for (int i = 0; i<numVars+1; i++) {
         double eigval = oes.eigenvalues()(i);
         if (abs(eigval) > 1.e-8) {
           Uo.col(index) = oes.eigenvectors().col(i)/pow(eigval, 0.5);
           index++;
           //cout << index<<"  "<<eigval<<endl;
         }
       }
       Uo.conservativeResize(numVars+1, index);

       MatrixXd Hessian_prime = Uo.transpose()*(Hessian * Uo);
       //cout << Hessian_prime <<endl;
       EigenSolver<MatrixXd> es(Hessian_prime);

       double emin=1.e10; int eminIndex;
       for (int i=0; i<index; i++) {

         if (es.eigenvalues()(i).real() < emin) {
           emin = es.eigenvalues()(i).real();
           eminIndex = i;
         }
       }
       
       VectorXcd update = Uo * es.eigenvectors().col(eminIndex);
       //if (commrank == 0) { 
       //cout << "Expected energy in next step :" << emin<<endl;
       //cout << "Number of non-redundant vars :" << index<<endl;
       //}
       
       for (int i = 0; i < vars.rows(); i++)
       {
         vars(i) += (update(i+1).real() / update(0).real());
       }
       
#ifndef SERIAL
       MPI_Bcast(&(vars[0]), vars.rows(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
       if (commrank == 0)
         std::cout << format("%5i %14.8f (%8.2e) %14.8f %8.1f %10i %8.2f\n") % iter % E0 % stddev % (grad.norm()) % (rt) % (schd.stochasticIter) % ((getTime() - startofCalc));
       iter++;

     }
   }
};   

#endif
