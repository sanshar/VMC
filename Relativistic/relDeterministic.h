#ifndef relDETERMINISTIC_HEADER_H
#define relDETERMINISTIC_HEADER_H
#include <Eigen/Dense>
#include <vector>
#include "relDeterminants.h"
#include "relWorkingArray.h"
#include "statistics.h"
#include "sr.h"
#include "linearMethod.h"
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
class relDeterministic
{
  public:
  Wfn &w;
  Walker &walk;
  long numVars;
  int norbs, nalpha, nbeta;
  std::vector<relDeterminant> allDets;
  relWorkingArray work;
  std::complex<double> ovlp, Eloc;
  double Overlap;
  Eigen::VectorXd grad_ratio, grad_Eloc;

  relDeterministic(Wfn &_w, Walker &_walk) : w(_w), walk(_walk)
  {
    numVars = w.getNumVariables();
    norbs = relDeterminant::norbs;
    nalpha = relDeterminant::nalpha;
    nbeta = relDeterminant::nbeta;
    relGenerateAllDeterminants(allDets, norbs, nalpha + nbeta);
    //generateAllDeterminants(allDets, norbs, nalpha, nbeta); //EDIT UNDO
    Overlap = 0.0;
  }
   
  void LocalEnergy(relDeterminant &D)
  {
    ovlp = 0.0 + 0.0i, Eloc = 0.0 + 0.0i;
    w.initWalker(walk, D);
    w.HamAndOvlp(walk, ovlp, Eloc, work, false);  
    if (0==1) { // && abs(ovlp)>1.0e-06) {
      //cout << walk << endl;
      cout << "Det " << D << " Eloc  " << Eloc << "  ovlp  " << ovlp << endl;
    }
  }
  
  void UpdateEnergy(std::complex<double> &Energy)
  {
    //cout << "in Update: ham  " << Eloc << "  ovlp  " << ovlp << " Overlap before: " << Overlap << " Energy before: " << Energy << endl;
    Overlap += std::abs(ovlp) * std::abs(ovlp); // EDIT: abs should be taken if imag
    //cout << "Overlap " << Overlap << "  ovlp^2 imag " << (ovlp * ovlp).imag() << endl << endl;
    //cout << "Energy " << Energy << "  Eloc " << Eloc << endl << endl;
    Energy += (Eloc * std::abs(ovlp) * std::abs(ovlp)); //EDIT: here take real part for real energies
  }

  void FinishEnergy(std::complex<double> &Energy)
  {
#ifndef SERIAL
    MPI_Allreduce(MPI_IN_PLACE, &(Overlap), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &(Energy), 1, MPI_C_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
#endif
    //cout << "in Finish: energy: " << Energy << " Overlap: " << Overlap << endl;
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
    w.OverlapWithLocalEnergyGradient(walk, work, grad_Eloc);
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

  void UpdateLM(DirectLM &H)
  {
    Eigen::VectorXd Gappended(numVars + 1), Happended(numVars + 1), Htemp(numVars);
    Gappended << 0.0, grad_ratio;
    Htemp = grad_Eloc + Eloc * grad_ratio;
    Happended << 0.0, Htemp;
    H.H.push_back(Happended);
    H.G.push_back(Gappended);
    H.T.push_back(ovlp * ovlp);
    H.Eloc.push_back(Eloc);
  }

  void UpdateLM(Eigen::MatrixXd &Hmatrix, Eigen::MatrixXd &Smatrix)
  {
    Eigen::VectorXd Gappended(numVars + 1), Happended(numVars + 1), Htemp(numVars);
    Gappended << 1.0, grad_ratio;
    Htemp = grad_Eloc + Eloc * grad_ratio;
    Happended << Eloc, Htemp;
    Smatrix += (ovlp * ovlp) * (Gappended * Gappended.adjoint());
    Hmatrix += (ovlp * ovlp) * (Gappended * Happended.adjoint());
  }

  void FinishLM(Eigen::MatrixXd &Hmatrix, Eigen::MatrixXd &Smatrix)
  {
#ifndef SERIAL
      MPI_Allreduce(MPI_IN_PLACE, (Smatrix.data()), Smatrix.rows() * Smatrix.cols(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, (Hmatrix.data()), Hmatrix.rows() * Hmatrix.cols(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
      Smatrix /= Overlap;
      Hmatrix /= Overlap;
  }
};
#endif
