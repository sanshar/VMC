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

     double E0, stddev, rt;
     while (iter < maxIter)
     {
       E0 = 0.0;
       stddev = 0.0;
       rt = 0.0;
       grad.setZero(numVars);
       x.setZero(numVars + 1);

       double acceptedFrac = getHessian(vars, grad, Hessian, Smatrix, E0, stddev, rt);
       

       write(vars);

       for (int i=0; i<numVars+1; i++)
         Hessian(i,i) += schd.hDiagShift;

       MatrixXd Uo = MatrixXd::Zero(numVars+1, numVars+1);       
       Uo(0,0) = 1.0;
       for (int i=0; i<numVars; i++) {
         Uo(0, i+1) = -Smatrix(0, i+1);
         Uo(i+1, i+1) = 1.0;
       }
       
       Smatrix = Uo.transpose()*(Smatrix * Uo);
       Hessian = Uo.transpose()*(Hessian * Uo);
/*
       ofstream Hout("H.txt");
       ofstream Sout("S.txt");
       for (int i = 0; i < Hessian.rows(); i++)
       {
         for (int j = 0; j < Hessian.cols(); j++)
         {
           Hout << Hessian(i, j) << "\t";
           Sout << Smatrix(i, j) << "\t";
         }
         Hout << endl;
         Sout << endl;
       }
       Hout.close();
       Sout.close();
*/
       SelfAdjointEigenSolver<MatrixXd> oes(Smatrix);

       int index = 0;
       Uo.setZero();
       for (int i = 0; i<numVars+1; i++) {
         double eigval = oes.eigenvalues()(i);
         if (abs(eigval) > 1.e-8) {
           Uo.col(index) = oes.eigenvectors().col(i)/pow(eigval, 0.5);
           index++;
           //cout << " | " << index<<"  "<<eigval<<endl;
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
       //cout << es.eigenvalues().transpose().real() << endl << endl;
       
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
         std::cout << format("%5i %14.8f (%8.2e) %14.8f %8.1f %8.1f %10i %8.2f\n") % iter % E0 % stddev % (grad.norm()) % (rt) % (acceptedFrac) % (schd.stochasticIter) % ((getTime() - startofCalc));
       iter++;

     }
   }
};   

class DirectLM
{
  public:
  std::vector<double> T, Eloc;
  std::vector<Eigen::VectorXd> G, H;
  double sdiagshift, hdiagshift;
  MatrixXd Smatrix, Hmatrix, GJD;
  DirectLM(double _hdiagshift, double _sdiagshift) : hdiagshift(_hdiagshift), sdiagshift(_sdiagshift) {}

  void BuildMatrices()
  {
    double Tau = 0.0;
    double Energy = 0.0;
    int dim = G[0].rows();
    Smatrix = MatrixXd::Zero(dim, dim);
    Hmatrix = MatrixXd::Zero(dim, dim);
    VectorXd H_bar = VectorXd::Zero(dim);
    VectorXd G_bar = VectorXd::Zero(dim);
    VectorXd G_Eloc_bar = VectorXd::Zero(dim);
    for (int i = 0; i < T.size(); i++)
    {
      Tau += T[i];
      Energy += T[i] * (Eloc[i] - Energy) / Tau;
      Smatrix += T[i] * (G[i] * G[i].adjoint() - Smatrix) / Tau;
      Hmatrix += T[i] * (G[i] * H[i].adjoint() - Hmatrix) / Tau;
      G_bar += T[i] * (G[i] - G_bar) / Tau;
      G_Eloc_bar += T[i] * (G[i] * Eloc[i] - G_Eloc_bar) / Tau;
      H_bar += T[i] * (H[i] - H_bar) / Tau;
    }
#ifndef SERIAL
      MPI_Allreduce(MPI_IN_PLACE, &(Energy), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(G_bar(0)), G_bar.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(G_Eloc_bar(0)), G_Eloc_bar.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(H_bar(0)), H_bar.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(Smatrix(0,0)), Smatrix.rows() * Smatrix.cols(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(Hmatrix(0,0)), Hmatrix.rows() * Hmatrix.cols(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
      Energy /= commsize;
      H_bar /= commsize;
      G_bar /= commsize;
      G_Eloc_bar /= commsize;
      Smatrix /= commsize;
      Hmatrix /= commsize;
      Smatrix = Smatrix - G_bar * G_bar.adjoint();
      Smatrix(0, 0) += 1.0;
      Hmatrix = Hmatrix - G_Eloc_bar * G_bar.adjoint() - G_bar * H_bar.adjoint() + Energy * G_bar * G_bar.adjoint();
      Hmatrix(0, 0) += Energy;
      Hmatrix.row(0) += (H_bar - Energy * G_bar).adjoint();
      Hmatrix.col(0) += G_Eloc_bar - Energy * G_bar;
  }

  void multiplyS(const VectorXd &x, VectorXd &Sx) const
  {
    double Tau = 0.0;
    int dim = x.rows();
    Sx.setZero(dim);
    VectorXd G_bar = VectorXd::Zero(dim);
    for (int i = 0; i < G.size(); i++)
    {
      Tau += T[i];
      Sx += T[i] * (G[i] * G[i].dot(x) - Sx) / Tau;
      G_bar += T[i] * (G[i] - G_bar) / Tau;
    }
#ifndef SERIAL
      MPI_Allreduce(MPI_IN_PLACE, &(Sx(0)), Sx.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(G_bar(0)), G_bar.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
      Sx /= commsize;
      G_bar /= commsize;
      Sx -= G_bar * G_bar.dot(x);
      Sx(0) += x(0); 
      Sx += sdiagshift * x;
  } 

  void multiplyH(const VectorXd &x, VectorXd &Hx) const
  {
    double Tau = 0.0;
    double Energy = 0.0;
    int dim = x.rows();
    Hx.setZero(dim);
    VectorXd G_bar = VectorXd::Zero(dim);
    VectorXd G_Eloc_bar = VectorXd::Zero(dim);
    VectorXd H_bar = VectorXd::Zero(dim);
    for (int i = 0; i < G.size(); i++)
    {
      Tau += T[i];
      Energy += T[i] * (Eloc[i] - Energy) / Tau;
      Hx += T[i] * (G[i] * H[i].dot(x) - Hx) / Tau;
      G_bar += T[i] * (G[i] - G_bar) / Tau;
      G_Eloc_bar += T[i] * (G[i] * Eloc[i] - G_Eloc_bar) / Tau;
      H_bar += T[i] * (H[i] - H_bar) / Tau;
    }
#ifndef SERIAL
      MPI_Allreduce(MPI_IN_PLACE, &(Energy), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(G_bar(0)), G_bar.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(G_Eloc_bar(0)), G_Eloc_bar.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(H_bar(0)), H_bar.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(Hx(0)), Hx.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
      Energy /= commsize;
      H_bar /= commsize;
      G_bar /= commsize;
      G_Eloc_bar /= commsize;
      Hx /= commsize;
      Hx -= G_Eloc_bar * G_bar.dot(x);
      Hx -= G_bar * H_bar.dot(x);
      Hx += Energy * G_bar * G_bar.dot(x);
      Hx(0) += Energy * x(0);
      Hx(0) += (H_bar - Energy * G_bar).dot(x); 
      Hx += (G_Eloc_bar - Energy * G_bar) * x(0); 
      Hx += hdiagshift * x;
  } 

  void multiplyH_thetaS(const VectorXd &x, double theta, VectorXd &Ax) const //Ax = (H - theta * S)x
  {
    double Tau = 0.0;
    double Energy = 0.0;
    int dim = x.rows();
    VectorXd Sx, Hx;
    Sx.setZero(dim);
    Hx.setZero(dim);
    VectorXd G_bar = VectorXd::Zero(dim);
    VectorXd G_Eloc_bar = VectorXd::Zero(dim);
    VectorXd H_bar = VectorXd::Zero(dim);
    for (int i = 0; i < G.size(); i++)
    {
      Tau += T[i];
      Energy += T[i] * (Eloc[i] - Energy) / Tau;
      //Sx
      Sx += T[i] * (G[i] * G[i].dot(x) - Sx) / Tau;
      //Hx
      Hx += T[i] * (G[i] * H[i].dot(x) - Hx) / Tau;

      G_bar += T[i] * (G[i] - G_bar) / Tau;
      G_Eloc_bar += T[i] * (G[i] * Eloc[i] - G_Eloc_bar) / Tau;
      H_bar += T[i] * (H[i] - H_bar) / Tau;
    }
#ifndef SERIAL
      MPI_Allreduce(MPI_IN_PLACE, &(Energy), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(G_bar(0)), G_bar.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(G_Eloc_bar(0)), G_Eloc_bar.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(H_bar(0)), H_bar.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(Hx(0)), Hx.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(Sx(0)), Sx.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
      Energy /= commsize;
      H_bar /= commsize;
      G_bar /= commsize;
      G_Eloc_bar /= commsize;
      Hx /= commsize;
      Sx /= commsize;
      Sx -= G_bar * G_bar.dot(x);
      Sx(0) += x(0); 
      Sx += sdiagshift * x;
      Hx -= G_Eloc_bar * G_bar.dot(x);
      Hx -= G_bar * H_bar.dot(x);
      Hx += Energy * G_bar * G_bar.dot(x);
      Hx(0) += Energy * x(0);
      Hx(0) += (H_bar - Energy * G_bar).dot(x); 
      Hx += (G_Eloc_bar - Energy * G_bar) * x(0); 
      Hx += hdiagshift * x;
      Ax = Hx - theta * Sx;
  }

  void multiplyGJD(const VectorXd &x, double theta, const VectorXd &u, VectorXd &Ax) const
  {
    //1st projector
    VectorXd a, b;
    multiplyS(x, a);
    b = x - u * u.dot(a);
    //(H - theta * S)
    multiplyH_thetaS(b, theta, a); 
    //2nd projector
    Ax = a;
    b = u * u.dot(a);
    multiplyS(b, a);
    Ax = Ax - a;
  } 
 
  void BuildGJD(double theta, const VectorXd &u)
  {
    BuildMatrices();
    Eigen::MatrixXd X = Eigen::MatrixXd::Identity(u.rows(), u.rows());
    Eigen::MatrixXd Xleft = X - Smatrix * u * u.adjoint();
    Eigen::MatrixXd Xright = X - u * u.adjoint() * Smatrix;
    GJD = Xleft * (Hmatrix - theta * Smatrix) * Xright;
  }
};


void generalizedJacobiDavidson(const Eigen::MatrixXd &H, const Eigen::MatrixXd &S, double &lambda, Eigen::VectorXd &v);

void GeneralizedJacobiDavidson(DirectLM &H, double target, const Eigen::VectorXd &targetv, double &lambda, Eigen::VectorXd &v, int n, double tol);

class directLM
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

    directLM(int pmaxIter=1000) : maxIter(pmaxIter)
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
     double rt = 0.0;
     int numVars = vars.rows();
     while (iter < maxIter)
     {
       double E0 = 0.0;
       double stddev = 0.0;
       VectorXd grad = VectorXd::Zero(numVars);
       DirectLM h(schd.hDiagShift, schd.sDiagShift);

       if (iter < schd.sgdIter)
         rt = 0.0;
       getHessian(vars, grad, h, E0, stddev, rt);
       write(vars);
       auto VMC_time = (getTime() - startofCalc);

       if (iter < schd.sgdIter)
       {
         vars += -0.1 * grad;
       }
       else
       {
         double lambda;
         VectorXd x(numVars + 1);
         VectorXd guess(numVars + 1);
         guess << 1.0, -(0.01 * grad);
         GeneralizedJacobiDavidson(h, E0, guess, lambda, x, schd.cgIter, schd.tol);
         for (int i = 0; i < vars.rows(); i++)
         {
           double dP = x(i + 1) / x(0);
           vars(i) += schd.stepsize * dP;
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
