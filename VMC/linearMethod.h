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


class LM
{
  private:
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
        ar & iter;
    }

  public:
    int iter;
    int maxIter;
    std::vector<double> stepsizes;
    double decay, hdiagshift, sdiagshift;

    LM(int _maxIter, std::vector<double> _stepsizes, double _hdiagshift, double _sdiagshift, double _decay) : maxIter(_maxIter), hdiagshift(_hdiagshift), sdiagshift(_sdiagshift), stepsizes(_stepsizes), decay(_decay)
    {
        iter = 0;
    }

    void write(VectorXd& vars)
    {
        if (commrank == 0)
        {
	    char file[50];
            sprintf(file, "LM.bkp");
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
            sprintf(file, "LM.bkp");
            std::ifstream ifs(file, std::ios::binary);
	        boost::archive::binary_iarchive load(ifs);
            load >> *this;
            load >> vars;
            ifs.close();
        }
    }

   template<typename Function1, typename Function2>
   void optimize(VectorXd &vars, Function1& getHessian, Function2 &runCorrelatedSampling, bool restart)
   {
     int numVars = vars.rows();
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

     VectorXd grad;
     MatrixXd Smatrix, Hessian;
     double E0, stddev, rt = 1.0;
     while (iter < maxIter)
     {
       E0 = 0.0;
       stddev = 0.0;
       grad.setZero(numVars);

       double acceptedFrac = getHessian(vars, grad, Hessian, Smatrix, E0, stddev, rt);
       write(vars);
       double VMC_time = (getTime() - startofCalc);

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


       double shift = hdiagshift * std::pow(decay, iter);
       if (shift < 1.e-8) shift = 1.e-8;
       for (int i=0; i<numVars+1; i++) {
         Hessian(i,i) += shift;
         Smatrix(i,i) += sdiagshift;
       }

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
         if (std::abs(eigval) > 1.e-8) {
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
       
       VectorXd x = (Uo * es.eigenvectors().col(eminIndex)).real();
       //VectorXd update = x.tail(numVars) / x(0);
         //normalize parameter update
         double ksi = 0.5;
         VectorXd Sx = Smatrix * x;
         double xSx = x.dot(Sx);
         VectorXd N = -((1.0 - ksi) * Sx) / ((1.0 - ksi) + (ksi * std::sqrt(xSx)));
         double norm = 1.0 - N.tail(numVars).dot(x.tail(numVars));
         VectorXd update = x.tail(numVars) / (x(0) * norm);
         double LM_time = (getTime() - startofCalc);
       //if (commrank == 0) { 
       //cout << "Expected energy in next step :" << emin<<endl;
       //cout << "Number of non-redundant vars :" << index<<endl;
       //}
         //correlated sampling
         std::vector<Eigen::VectorXd> V(stepsizes.size(), vars);
         std::vector<double> E(stepsizes.size(), 0.0);
         for (int i = 0; i < V.size(); i++)
         {
           V[i] += stepsizes[i] * update;
         }
         runCorrelatedSampling(V, E);
         index = 0;
         for (int i = 0; i < E.size(); i++)
         {
           if (E[i] < E[index])
             index = i;              
         }
         //vars = V[index];

         if (schd.printOpt && commrank == 0)
         {
           cout << "Correlated Sampling: " << endl;
           for (int i = 0; i < E.size(); i++)
           {
             cout << stepsizes[i] << " " << E[i] << "|";
           }
           cout << endl;
         }

         //accept update only if "good" move
         if (((E[index] <= (E0 + stddev)) && (E[index] >= (E0 - 1.0))) || iter < 2)
         {
           //if (commrank == 0) cout << "update accepted" << endl;
           vars = V[index];
         }
#ifndef SERIAL
       MPI_Bcast(&(vars[0]), vars.rows(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
       if (commrank == 0)
         std::cout << format("%5i %14.8f (%8.2e) %14.8f %8.1f %8.1f %10i %8.2f %8.2f %8.2f\n") % iter % E0 % stddev % (grad.norm()) % (rt) % (acceptedFrac) % (schd.stochasticIter) % (VMC_time) % (LM_time) % ((getTime() - startofCalc));
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
  DirectLM() {}
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
      Energy += T[i] * Eloc[i];
      Smatrix += T[i] * (G[i] * G[i].adjoint());
      Hmatrix += T[i] * (G[i] * H[i].adjoint());
      G_bar += T[i] * G[i];
      G_Eloc_bar += T[i] * (G[i] * Eloc[i]);
      H_bar += T[i] * H[i];
    }
#ifndef SERIAL
      MPI_Allreduce(MPI_IN_PLACE, &(Energy), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(Tau), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(G_bar(0)), G_bar.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(G_Eloc_bar(0)), G_Eloc_bar.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(H_bar(0)), H_bar.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(Smatrix(0,0)), Smatrix.rows() * Smatrix.cols(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(Hmatrix(0,0)), Hmatrix.rows() * Hmatrix.cols(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
      Energy /= Tau;
      H_bar /= Tau;
      G_bar /= Tau;
      G_Eloc_bar /= Tau;
      Smatrix /= Tau;
      Hmatrix /= Tau;
      Smatrix = Smatrix - G_bar * G_bar.adjoint();
      Smatrix(0, 0) += 1.0;
      Hmatrix = Hmatrix - G_Eloc_bar * G_bar.adjoint() - G_bar * H_bar.adjoint() + Energy * G_bar * G_bar.adjoint();
      Hmatrix(0, 0) += Energy;
      Hmatrix.row(0) += (H_bar - Energy * G_bar).adjoint();
      Hmatrix.col(0) += G_Eloc_bar - Energy * G_bar;
  }

  void multiplyS(const VectorXd &x, VectorXd &Sx) const
  {
    VectorXd xcopy(x);
    xcopy(0) = 0.0;
    double Tau = 0.0;
    int dim = x.rows();
    Sx.setZero(dim);
    VectorXd G_bar = VectorXd::Zero(dim);
    for (int i = 0; i < G.size(); i++)
    {
      Tau += T[i];
      Sx += T[i] * (G[i] * G[i].dot(x));
      G_bar += T[i] * G[i];
    }
#ifndef SERIAL
      MPI_Allreduce(MPI_IN_PLACE, &(Tau), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(Sx(0)), Sx.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(G_bar(0)), G_bar.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
      Sx /= Tau;
      G_bar /= Tau;
      Sx -= G_bar * G_bar.dot(x);
      Sx(0) += x(0); 
      Sx += sdiagshift * xcopy;
  } 

  void multiplyH(const VectorXd &x, VectorXd &Hx) const
  {
    VectorXd xcopy(x);
    xcopy(0) = 0.0;
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
      Energy += T[i] * Eloc[i];
      Hx += T[i] * (G[i] * H[i].dot(x));
      G_bar += T[i] * G[i];
      G_Eloc_bar += T[i] * (G[i] * Eloc[i]);
      H_bar += T[i] * H[i];
    }
#ifndef SERIAL
      MPI_Allreduce(MPI_IN_PLACE, &(Energy), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(Tau), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(G_bar(0)), G_bar.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(G_Eloc_bar(0)), G_Eloc_bar.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(H_bar(0)), H_bar.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(Hx(0)), Hx.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
      Energy /= Tau;
      H_bar /= Tau;
      G_bar /= Tau;
      G_Eloc_bar /= Tau;
      Hx /= Tau;
      Hx -= G_Eloc_bar * G_bar.dot(x);
      Hx -= G_bar * H_bar.dot(x);
      Hx += Energy * G_bar * G_bar.dot(x);
      Hx(0) += Energy * x(0);
      Hx(0) += (H_bar - Energy * G_bar).dot(x); 
      Hx += (G_Eloc_bar - Energy * G_bar) * x(0); 
      Hx += hdiagshift * xcopy;
  } 

  void multiplyH_thetaS(const VectorXd &x, double theta, VectorXd &Ax) const //Ax = (H - theta * S)x
  {
    VectorXd xcopy(x);
    xcopy(0) = 0.0;
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
      Energy += T[i] * Eloc[i];
      //Sx
      Sx += T[i] * (G[i] * G[i].dot(x));
      //Hx
      Hx += T[i] * (G[i] * H[i].dot(x));

      G_bar += T[i] * G[i];
      G_Eloc_bar += T[i] * (G[i] * Eloc[i]);
      H_bar += T[i] * H[i];
    }
#ifndef SERIAL
      MPI_Allreduce(MPI_IN_PLACE, &(Energy), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(Tau), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(G_bar(0)), G_bar.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(G_Eloc_bar(0)), G_Eloc_bar.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(H_bar(0)), H_bar.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(Hx(0)), Hx.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(Sx(0)), Sx.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
      Energy /= Tau;
      H_bar /= Tau;
      G_bar /= Tau;
      G_Eloc_bar /= Tau;
      Hx /= Tau;
      Sx /= Tau;
      Sx -= G_bar * G_bar.dot(x);
      Sx(0) += x(0); 
      Sx += sdiagshift * xcopy;
      Hx -= G_Eloc_bar * G_bar.dot(x);
      Hx -= G_bar * H_bar.dot(x);
      Hx += Energy * G_bar * G_bar.dot(x);
      Hx(0) += Energy * x(0);
      Hx(0) += (H_bar - Energy * G_bar).dot(x); 
      Hx += (G_Eloc_bar - Energy * G_bar) * x(0); 
      Hx += hdiagshift * xcopy;
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

void GeneralizedJacobiDavidson(DirectLM &H, double target, const Eigen::VectorXd &targetv, double &lambda, Eigen::VectorXd &v, int cgIter, double cgTol, double dTol);

class directLM
{
  private:
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
        ar & iter & rt & mom1 & mom2;
    }

  public:
    int iter;
    int maxIter;
    int numLMiter;
    double rt;
    std::vector<double> LMStep;
    double hdiagshift, sdiagshift, LMDecay;
    int AMSGradIter;
    double AMSGradStep, AMSGradDecay1, AMSGradDecay2;
    VectorXd mom1;
    VectorXd mom2;

    directLM(int _maxIter, std::vector<double> _LMStep, double _hdiagshift, double _sdiagshift, double _LMDecay, int _AMSGradIter = 100, double _AMSGradStep = 0.001, double _AMSGradDecay1 = 0.1, double _AMSGradDecay2 = 0.001) : maxIter(_maxIter), LMStep(_LMStep), hdiagshift(_hdiagshift), sdiagshift(_sdiagshift), LMDecay(_LMDecay), AMSGradIter(_AMSGradIter), AMSGradStep(_AMSGradStep), AMSGradDecay1(_AMSGradDecay1), AMSGradDecay2(_AMSGradDecay2)
    {
        iter = 0;
        numLMiter = 0;
        rt = 1.0;
    }

    void write(VectorXd& vars)
    {
        if (commrank == 0)
        {
	        char file[50];
            sprintf(file, "directLM.bkp");
            std::ofstream ofs(file, std::ios::binary);
            boost::archive::binary_oarchive save(ofs);
            save << *this;
            save << mom1;
            save << mom2;
            save << rt;
            save << vars;
            ofs.close();
        }
    }

    void read(VectorXd& vars)
    {
        if (commrank == 0)
        {
	        char file[50];
            sprintf(file, "directLM.bkp");
            std::ifstream ifs(file, std::ios::binary);
	        boost::archive::binary_iarchive load(ifs);
            load >> *this;
            load >> mom1;
            load >> mom2;
            load >> rt;
            load >> vars;
            ifs.close();
        }
    }

   template<typename Function1, typename Function2>
   void optimize(VectorXd &vars, Function1 &getHessian, Function2 &runCorrelatedSampling, bool restart)
   {
     int numVars = vars.rows();
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
     else if (mom1.rows() == 0)
     {
       mom1.setZero(numVars);
       mom2.setZero(numVars);
     }
     while (iter < maxIter)
     {
if (commrank == 0 && schd.printOpt) std::cout << "Iteration start" << endl;
       if (iter < AMSGradIter)
         rt = 0.0;
       double E0 = 0.0;
       double stddev = 0.0;
       VectorXd grad = VectorXd::Zero(numVars);
       double shift = hdiagshift * std::pow(LMDecay, numLMiter);
       if (shift < 1.e-8) shift = 1.e-8;
       DirectLM h(shift, sdiagshift);

       double acceptedFrac = getHessian(vars, grad, h, E0, stddev, rt);
       write(vars);

if (commrank == 0 && schd.printOpt) std::cout << "VMC run complete" << endl;
if (commrank == 0 && schd.printOpt) std::cout << "LM shift: " << shift << endl;
       double VMC_time = (getTime() - startofCalc);
       double LM_time = VMC_time;

       if (iter < AMSGradIter)
       {
         for (int i = 0; i < vars.rows(); i++)
         {
           mom1(i) = AMSGradDecay1 * grad(i) + (1.0 - AMSGradDecay1) * mom1(i);
           mom2(i) = std::max(mom2(i), AMSGradDecay2 * grad(i) * grad(i) + (1.0 - AMSGradDecay2) * mom2(i));
           double delta = AMSGradStep * mom1(i) / (pow(mom2(i), 0.5) + 1.e-8);
           vars(i) -= delta;
         }
       }
       else
       {
         numLMiter++;
         double lambda;
         VectorXd x(numVars + 1);
         //VectorXd guess(numVars + 1);
         //guess << 1.0, -(0.01 * grad);
         VectorXd guess = VectorXd::Unit(numVars + 1, 0);
         GeneralizedJacobiDavidson(h, E0, guess, lambda, x, schd.cgIter, schd.cgTol, schd.dTol);
         //normalize parameter update
         double ksi = 0.5;
         VectorXd Sx;
         h.multiplyS(x, Sx);
         double xSx = x.dot(Sx);
         VectorXd N = -((1.0 - ksi) * Sx) / ((1.0 - ksi) + (ksi * std::sqrt(xSx)));
         double norm = 1.0 - N.tail(numVars).dot(x.tail(numVars));
         VectorXd update = x.tail(numVars) / (x(0) * norm);
         //VectorXd update = x.tail(numVars) / x(0);
if (commrank == 0 && schd.printOpt) std::cout << "LM complete" << endl;
         LM_time = (getTime() - startofCalc);
         //correlated sampling
         std::vector<Eigen::VectorXd> V(LMStep.size(), vars);
         std::vector<double> E(LMStep.size(), 0.0);
         for (int i = 0; i < V.size(); i++)
         {
           V[i] += LMStep[i] * update;
         }
         runCorrelatedSampling(V, E);
if (commrank == 0 && schd.printOpt) std::cout << "CorrSample complete" << endl;
         int index = 0;
         for (int i = 0; i < E.size(); i++)
         {
           if (E[i] < E[index])
             index = i;              
         }
         //vars = V[index];
         
         if (schd.printOpt && commrank == 0)
         {
           cout << "Correlated Sampling: " << endl;
           for (int i = 0; i < E.size(); i++)
           {
             cout << LMStep[i] << " " << E[i] << " | ";
           }
           cout << endl;
         }

         //accept update only if "good" move
         if (((E[index] <= (E0 + stddev)) && (E[index] >= (E0 - 1.0))) || iter < 2)
         {
           //if (commrank == 0) cout << "update accepted" << endl;
           vars = V[index];
         }
       } 
#ifndef SERIAL
       MPI_Bcast(&(vars[0]), vars.rows(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
       if (commrank == 0)
         std::cout << format("%5i %14.8f (%8.2e) %14.8f %8.1f %8.1f %8.2f %8.2f %8.2f \n") % iter % E0 % stddev % (grad.norm()) % (rt) % (acceptedFrac) % (VMC_time) % (LM_time) % ((getTime() - startofCalc));
       iter++;
     }
   }
};   

/*
class DirectVarLM : public DirectLM
{
    public:
    DirectVarLM(double _hdiagshift, double _sdiagshift) : DirectLM(_hdiagshift, _sdiagshift) {}

    void BuildMatrices()
    {
      int dim = G[0].rows();
      double Tau = 0.0;
      double Energy = 0.0;
      double Variance = 0.0;
      Smatrix = MatrixXd::Zero(dim, dim);
      Hmatrix = MatrixXd::Zero(dim, dim);
      VectorXd H_bar = VectorXd::Zero(dim);
      VectorXd G_bar = VectorXd::Zero(dim);
      VectorXd grad_Energy = VectorXd::Zero(dim);
      VectorXd grad_Variance = VectorXd::Zero(dim);
      for (int i = 0; i < T.size(); i++)
      {
        Tau += T[i];
        Energy += T[i] * (Eloc[i] - Energy) / Tau;
        Variance += T[i] * (Eloc[i] * Eloc[i] - Variance) / Tau;
        Smatrix += T[i] * (G[i] * G[i].adjoint() - Smatrix) / Tau;
        Hmatrix += T[i] * (H[i] * H[i].adjoint() - Hmatrix) / Tau;
        G_bar += T[i] * (G[i] - G_bar) / Tau;
        H_bar += T[i] * (H[i] - H_bar) / Tau;
        grad_Energy += T[i] * (G[i] * Eloc[i] - grad_Energy) / Tau;
        grad_Variance += T[i] * (Eloc[i] * H[i] - grad_Variance) / Tau;
      }
#ifndef SERIAL
      MPI_Allreduce(MPI_IN_PLACE, &(Energy), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(Variance), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(G_bar(0)), G_bar.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(H_bar(0)), H_bar.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(Smatrix(0,0)), Smatrix.rows() * Smatrix.cols(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(Hmatrix(0,0)), Hmatrix.rows() * Hmatrix.cols(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(grad_Energy(0)), grad_Energy.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(grad_Variance(0)), grad_Variance.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
      Energy /= commsize;
      Variance /= commsize;
      Variance -= Energy * Energy;
      H_bar /= commsize;
      G_bar /= commsize;
      Smatrix /= commsize;
      Hmatrix /= commsize;
      grad_Energy /= commsize;
      grad_Variance /= commsize;

      grad_Energy = 2.0 * (grad_Energy - G_bar * Energy);
      grad_Variance = grad_Variance - H_bar * Energy;
      Smatrix = Smatrix - G_bar * G_bar.adjoint();
      Hmatrix = Hmatrix - H_bar * grad_Energy.adjoint() - grad_Energy * H_bar.adjoint() + grad_Energy * grad_Energy.adjoint();
      Hmatrix += Variance * Smatrix;
      Smatrix(0, 0) += 1.0;
      Hmatrix(0, 0) += Variance;
      Hmatrix.row(0) += grad_Variance.adjoint();
      Hmatrix.col(0) += grad_Variance;
    }
  
    void multiplyH(const VectorXd &x, VectorXd &Hx) const
    {
      int dim = x.rows();
      VectorXd xcopy(x);
      xcopy(0) = 0.0;
      double Tau = 0.0;
      double Energy = 0.0;
      double Variance = 0.0;
      VectorXd Sx = VectorXd::Zero(dim);
      Hx.setZero(dim);
      VectorXd H_bar = VectorXd::Zero(dim);
      VectorXd G_bar = VectorXd::Zero(dim);
      VectorXd grad_Energy = VectorXd::Zero(dim);
      VectorXd grad_Variance = VectorXd::Zero(dim);
      for (int i = 0; i < T.size(); i++)
      {
        Tau += T[i];
        Energy += T[i] * (Eloc[i] - Energy) / Tau;
        Variance += T[i] * (Eloc[i] * Eloc[i] - Variance) / Tau;
        Sx += T[i] * (G[i] * G[i].dot(x) - Sx) / Tau;
        Hx += T[i] * (H[i] * H[i].dot(x) - Hx) / Tau;
        G_bar += T[i] * (G[i] - G_bar) / Tau;
        H_bar += T[i] * (H[i] - H_bar) / Tau;
        grad_Energy += T[i] * (G[i] * Eloc[i] - grad_Energy) / Tau;
        grad_Variance += T[i] * (Eloc[i] * H[i] - grad_Variance) / Tau;
      }
#ifndef SERIAL
      MPI_Allreduce(MPI_IN_PLACE, &(Energy), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(Variance), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(G_bar(0)), G_bar.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(H_bar(0)), H_bar.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(Sx(0)), Sx.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(Hx(0)), Hx.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(grad_Energy(0)), grad_Energy.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(grad_Variance(0)), grad_Variance.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
      Energy /= commsize;
      Variance /= commsize;
      Variance -= Energy * Energy;
      H_bar /= commsize;
      G_bar /= commsize;
      Sx /= commsize;
      Hx /= commsize;
      grad_Energy /= commsize;
      grad_Variance /= commsize;
      grad_Energy = 2.0 * (grad_Energy - G_bar * Energy);
      grad_Variance = grad_Variance - H_bar * Energy;
      Sx -= G_bar * G_bar.dot(x);
      Hx += - H_bar * grad_Energy.dot(x) - grad_Energy * H_bar.dot(x) + grad_Energy * grad_Energy.dot(x);
      Hx += Variance * Sx;
      Hx(0) += Variance * x(0);
      Hx(0) += grad_Variance.dot(x);
      Hx += grad_Variance * x(0);
      Hx += hdiagshift * xcopy;
    } 

    void multiplyH_thetaS(const VectorXd &x, double theta, VectorXd &Ax) const
    {
      int dim = x.rows();
      VectorXd xcopy(x);
      xcopy(0) = 0.0;
      double Tau = 0.0;
      double Energy = 0.0;
      double Variance = 0.0;
      VectorXd Sx = VectorXd::Zero(dim);
      VectorXd Hx = VectorXd::Zero(dim);
      VectorXd H_bar = VectorXd::Zero(dim);
      VectorXd G_bar = VectorXd::Zero(dim);
      VectorXd grad_Energy = VectorXd::Zero(dim);
      VectorXd grad_Variance = VectorXd::Zero(dim);
      for (int i = 0; i < T.size(); i++)
      {
        Tau += T[i];
        Energy += T[i] * (Eloc[i] - Energy) / Tau;
        Variance += T[i] * (Eloc[i] * Eloc[i] - Variance) / Tau;
        Sx += T[i] * (G[i] * G[i].dot(x) - Sx) / Tau;
        Hx += T[i] * (H[i] * H[i].dot(x) - Hx) / Tau;
        G_bar += T[i] * (G[i] - G_bar) / Tau;
        H_bar += T[i] * (H[i] - H_bar) / Tau;
        grad_Energy += T[i] * (G[i] * Eloc[i] - grad_Energy) / Tau;
        grad_Variance += T[i] * (Eloc[i] * H[i] - grad_Variance) / Tau;
      }
#ifndef SERIAL
      MPI_Allreduce(MPI_IN_PLACE, &(Energy), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(Variance), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(G_bar(0)), G_bar.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(H_bar(0)), H_bar.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(Sx(0)), Sx.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(Hx(0)), Hx.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(grad_Energy(0)), grad_Energy.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &(grad_Variance(0)), grad_Variance.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
      Energy /= commsize;
      Variance /= commsize;
      Variance -= Energy * Energy;
      H_bar /= commsize;
      G_bar /= commsize;
      Sx /= commsize;
      Hx /= commsize;
      grad_Energy /= commsize;
      grad_Variance /= commsize;
      grad_Energy = 2.0 * (grad_Energy - G_bar * Energy);
      grad_Variance = grad_Variance - H_bar * Energy;
      Sx -= G_bar * G_bar.dot(x);
      Hx += - H_bar * grad_Energy.dot(x) - grad_Energy * H_bar.dot(x) + grad_Energy * grad_Energy.dot(x);
      Hx += Variance * Sx;
      Hx(0) += Variance * x(0);
      Hx(0) += grad_Variance.dot(x);
      Hx += grad_Variance * x(0);
      Hx += hdiagshift * xcopy;
      Sx(0) += x(0);
      Sx += sdiagshift * xcopy;
      Ax = Hx - theta * Sx;
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
  double tol = 1.e-8;

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

class directVarLM
{
  private:
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
        ar & maxIter & iter & rt;
    }

  public:
    int maxIter;
    int iter;
    double rt;

    directVarLM(int pmaxIter=1000) : maxIter(pmaxIter)
    {
        iter = 0;
    }

    void write(VectorXd& vars)
    {
        if (commrank == 0)
        {
	    char file[5000];
            sprintf(file, "directVarLM.bkp");
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
            sprintf(file, "directVarLM.bkp");
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
     if (!restart)
       rt = 0.0;
     while (iter < maxIter)
     {
cout << "a" << endl;
       double E0 = 0.0;
       double variance = 0.0;
       double stddev = 0.0;
       VectorXd grad = VectorXd::Zero(numVars);
       int numLMiter = iter - schd.sgdIter;
       DirectVarLM H(schd.hDiagShift, schd.sDiagShift);

cout << "b" << endl;
       if (iter < schd.sgdIter)
         rt = 0.0;
       getVariance(vars, grad, H, variance, E0, stddev, rt);
cout << "Var: " << variance;
cout << "c" << endl;
       write(vars);
       auto VMC_time = (getTime() - startofCalc);

if (iter >= schd.sgdIter)
{
       H.BuildMatrices();
       cout << H.Hmatrix << endl;
       cout << H.Smatrix << endl;
       Eigen::GeneralizedSelfAdjointEigenSolver<MatrixXd> es(H.Hmatrix, H.Smatrix);
       cout << es.eigenvalues().adjoint() << endl;
}

       if (iter < schd.sgdIter)
       {
         vars += -0.1 * grad;
       }
       else 
       {
         double lambda;
         VectorXd x(numVars + 1);
         VectorXd guess = VectorXd::Unit(numVars + 1, 0);
         GeneralizedJacobiDavidson(H, variance, guess, lambda, x, schd.cgIter, schd.tol);
         for (int i = 0; i < x.rows(); i++)
         {
           double dP = x(i + 1) / x(0);
           vars(i) += schd.stepsize * dP;
         }
       }
cout << "d" << endl;
#ifndef SERIAL
       MPI_Bcast(&(vars[0]), vars.rows(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
       if (commrank == 0)
         std::cout << format("%5i %14.8f %14.8f (%8.2e) %14.8f %8.1f %8.2f %8.2f\n") % iter % variance % E0 % stddev % (grad.norm()) % (rt) % (VMC_time) % ((getTime() - startofCalc));
       iter++;
     }
   }
};   
*/
#endif
