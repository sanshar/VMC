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
  std::complex<double> ovlp, ovlpCurr, Eloc;
  double Overlap;
  Eigen::VectorXd grad_ratio, grad_Eloc;

  std::vector<std::complex<double>> ovlpLS;


  relDeterministic(Wfn &_w, Walker &_walk) : w(_w), walk(_walk)
  {
    numVars = w.getNumVariables();
    norbs = relDeterminant::norbs;
    nalpha = relDeterminant::nalpha;
    nbeta = relDeterminant::nbeta;
    relGenerateAllDeterminants(allDets, norbs, nalpha + nbeta);
    //generateAllDeterminants(allDets, norbs, nalpha, nbeta); //EDIT UNDO
    Overlap = 0.0;
    for (int i=0; i<schd.excitedState; i++) {  
      ovlpLS.push_back(i);
    }
  }
   
  void LocalOverlap(relDeterminant &D, std::vector<Wfn> &ls)
  {
    ovlpCurr = 0.0 + 0.0i;
    w.initWalker(walk, D);
    w.OvlpPenalty(walk, ovlpCurr);
    //cout << "Det " << D << "  ovlp  " << ovlp << "  ovlpLS  ";
    for (int i=0; i<schd.excitedState; i++) {  
      ovlpLS[i] = 0.0 + 0.0i;
      ls[i].initWalker(walk, D);
      ls[i].OvlpPenalty(walk, ovlpLS[i]);  
      if (0==1) { // && abs(ovlp)>1.0e-06) {
        cout << ovlpLS[i];
      }
    }
    //cout << endl;
  }

  void UpdateOverlap(std::vector<std::complex<double>> &ovlpPen)
  {
    for (int i=0; i<schd.excitedState; i++) {  
      ovlpPen[i] += (ovlpLS[i] / ovlpCurr) * std::abs(ovlp) * std::abs(ovlp);
    }
  }

  void FinishOverlap(std::vector<std::complex<double>> &ovlpPen)
  {
    for (int i=0; i<schd.excitedState; i++) {  
#ifndef SERIAL
      MPI_Allreduce(MPI_IN_PLACE, &(ovlpPen[i]), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
      ovlpPen[i] /= Overlap;
    }
  }

  void LocalEnergy(relDeterminant &D)
  {
    ovlp = 0.0 + 0.0i, Eloc = 0.0 + 0.0i;
    w.initWalker(walk, D);
    w.HamAndOvlp(walk, ovlp, Eloc, work, false);  
    if (0==1) { // && abs(ovlp)>1.0e-06) {
      cout << "Det " << D << " Eloc  " << Eloc << "  ovlp  " << ovlp << endl;
    }
  }
  
  void UpdateEnergy(std::complex<double> &Energy)
  {
    Overlap += std::abs(ovlp) * std::abs(ovlp); // EDIT: abs should be taken if imag
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
    grad += grad_ratio * Eloc.real() * std::abs(ovlp) * std::abs(ovlp);
    grad_ratio_bar += grad_ratio * std::abs(ovlp) * std::abs(ovlp);
  }

  void UpdateGradientPenalty(std::vector<Eigen::VectorXd> &gradPen, const std::vector<std::complex<double>> &ovlpPen)
  {
    for (int i=0; i<schd.excitedState; i++) {
      double sgn = 1.0;
      if (std::abs(ovlpPen[i])<0) sgn=-1.0;
      gradPen[i] += sgn* grad_ratio * std::abs(ovlpPen[i]) * std::abs(ovlp) * std::abs(ovlp);
    }
  }


  void FinishGradient(Eigen::VectorXd &grad, Eigen::VectorXd &grad_ratio_bar, const std::complex<double> &Energy)
  {
#ifndef SERIAL
    MPI_Allreduce(MPI_IN_PLACE, (grad_ratio_bar.data()), grad_ratio_bar.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, (grad.data()), grad_ratio.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
    grad = (grad - Energy.real() * grad_ratio_bar) / Overlap;
  }



  void FinishGradientPenalty(std::vector<Eigen::VectorXd> &gradPen, const Eigen::VectorXd &grad_ratio_bar, const std::vector<std::complex<double>> &ovlpPen)
  {
    for (int i=0; i<schd.excitedState; i++) {
#ifndef SERIAL
      MPI_Allreduce(MPI_IN_PLACE, (gradPen[i].data()), grad_ratio.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
      gradPen[i] = (gradPen[i] - 2*std::abs(ovlpPen[i]) * grad_ratio_bar) / Overlap;
      gradPen[i] *= 3;
    }
  }


  void CombineGradients(Eigen::VectorXd &grad, std::vector<Eigen::VectorXd> &gradPen)
  {
    for (int i=0; i<schd.excitedState; i++) {
      grad += gradPen[i];
    }
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
