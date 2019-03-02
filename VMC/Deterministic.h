#ifndef DETERMINISTIC_HEADER_H
#define DETERMINISTIC_HEADER_H
#include <Eigen/Dense>
#include <vector>
#include "Determinants.h"
#include "workingArray.h"
#include "statistics.h"
#include "sr.h"
#include "evaluateE.h"
#include "global.h"
#include <iostream>
#include <fstream>
#include <boost/serialization/serialization.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <algorithm>

#ifndef SERIAL
#include "mpi.h"
#endif

template<typename Wfn, typename Walker>
class Deterministic
{
  public:
  Wfn w;
  Walker walk;
  long numVars;
  int norbs, nalpha, nbeta;
  std::vector<Determinant> allDets;
  workingArray work;
  double ovlp, Eloc;
  double Overlap;
  Eigen::VectorXd grad_ratio;

  Deterministic(Wfn _w, Walker _walk) : w(_w), walk(_walk)
  {
    numVars = w.getNumVariables();
    norbs = Determinant::norbs;
    nalpha = Determinant::nalpha;
    nbeta = Determinant::nbeta;
    generateAllDeterminants(allDets, norbs, nalpha, nbeta);
    Overlap = 0.0;
  }
   
  void LocalEnergy(Determinant &D)
  {
    ovlp = 0.0, Eloc = 0.0;
    w.initWalker(walk, D);
    w.HamAndOvlp(walk, ovlp, Eloc, work, false);  
    if (schd.debug) {
      cout << walk << endl;
      cout << "ham  " << Eloc << "  ovlp  " << ovlp << endl << endl;
    }
  }
  
  void UpdateEnergy(double &Energy)
  {
    Overlap += ovlp * ovlp;
    Energy += Eloc * ovlp * ovlp;
  }

  void FinishEnergy(double &Energy)
  {
#ifndef SERIAL
    MPI_Allreduce(MPI_IN_PLACE, &(Overlap), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &(Energy), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
    Energy /= Overlap;
  }

  void LocalGradient()
  {
    grad_ratio.setZero(numVars);
    w.OverlapWithGradient(walk, ovlp, grad_ratio);
  }

  void UpdateGradient(Eigen::VectorXd &grad, Eigen::VectorXd &grad_ratio_bar)
  {
    grad += grad_ratio * Eloc * ovlp * ovlp;
    grad_ratio_bar += grad_ratio * ovlp * ovlp;
  }

  void FinishGradient(Eigen::VectorXd &grad, Eigen::VectorXd &grad_ratio_bar, const double &Energy)
  {
#ifndef SERIAL
    MPI_Allreduce(MPI_IN_PLACE, (grad_ratio_bar.data()), grad_ratio_bar.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, (grad.data()), grad_ratio.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
    grad = (grad - Energy * grad_ratio_bar) / Overlap;
  }

  void LocalEnergyGradient()
  {
    LocalEnergySolver Solver(walk.d);
    Eigen::VectorXd vars;
    w.getVariables(vars);
    double ElocTest;
    stan::math::gradient(Solver, vars, ElocTest, grad_local_energy);
    
    //below is very expensive and used only for debugging
/*
    cout << Eloc << "\t|\t" << ElocTest << endl << endl;
    VectorXd finiteGradEloc = VectorXd::Zero(vars.size());
    for (int i = 0; i < vars.size(); ++i)
    {
      double dt = 0.00001;
      VectorXd varsdt = vars;
      varsdt(i) += dt;
      finiteGradEloc(i) = (Solver(varsdt) - ElocTest) / dt;
    }
    for (int i = 0; i < vars.size(); ++i)
    {
      cout << finiteGradEloc(i) << "\t" << grad_local_energy(i) << endl;
    }
*/
  }
  
  void UpdateSR(DirectMetric &S)
  {
    Eigen::VectorXd appended(numVars);
    appended << 1.0, grad_ratio;
    if (schd.direct)
    {
      S.Vectors.push_back(appended);
      S.T.push_back(ovlp * ovlp);
    }
    else
    {
      S.Smatrix += (ovlp * ovlp) * appended * appended.adjoint();
    }
  }
  
  void FinishSR(const Eigen::VectorXd &grad, const Eigen::VectorXd &grad_ratio_bar, Eigen::VectorXd &H, DirectMetric S)
  {
    H.setZero(numVars + 1);
    Eigen::VectorXd appended(numVars);
    appended = grad_ratio_bar - schd.stepsize * grad;
    H << 1.0, appended;
    if (!(schd.direct))
    {
#ifndef SERIAL
      MPI_Allreduce(MPI_IN_PLACE, (S.Smatrix.data()), S.Smatrix.rows() * S.Smatrix.cols(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
      S.Smatrix /= Overlap;
    }
  }
};
#endif
