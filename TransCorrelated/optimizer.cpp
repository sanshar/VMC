#include "optimizer.h"
#include "Residuals.h"
#include <iostream>
#include "DirectJacobian.h"
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>
#include "global.h"
#include <boost/format.hpp>
using namespace boost;
using namespace std;


void optimizeJastrowParams(
    VectorXd& params,
    boost::function<int (const VectorXd&, VectorXd&)>& func,
    Residuals& residual) {

  
  cout << "optimize jastrow "<<endl;
  double Energy, norm=10.;
  int iter = 0;
  VectorXd residue(params.size());
  
  DirectJacobian<double> J(func);
  Eigen::GMRES<DirectJacobian<double>, Eigen::IdentityPreconditioner> gmres;
  gmres.compute(J);
  gmres.setTolerance(1.e-4);
  func(params, residue);
  norm = residue.norm();
  
  std::cout << format("%5i   %14.8f   %14.6f %6.6f \n") %(iter) %(residual.E0)
      %(residue.norm()) %(getTime()-startofCalc);

  while(norm > 1.e-5) {      
    J.setFvec(params, residue);
    residue *=-1.;
    VectorXd x = gmres.solve(residue);
    params += x;
    iter++;
    
    func(params, residue);
    norm = residue.norm();
    
    std::cout << format("%5i   %14.8f   %14.6f %6.6f \n") %(iter) %(residual.E0)
        %(residue.norm()) %(getTime()-startofCalc);
  }
}


void optimizeOrbitalParams(
    VectorXd& params,
    boost::function<int (const VectorXd&, VectorXd&)>& func,
    Residuals& residual) {

  cout << "optimize orbitals "<<endl;
  double Energy, norm=10.;
  int iter = 0;
  VectorXd residue(params.size());
  
  DirectJacobian<double> J(func);
  Eigen::GMRES<DirectJacobian<double>, Eigen::IdentityPreconditioner> gmres;
  gmres.compute(J);
  gmres.setTolerance(1.e-4);

  func(params, residue);
  norm = residue.norm();

  std::cout << format("%5i   %14.8f   %14.6f %6.6f \n") %(iter) %(residual.E0)
      %(residue.norm()) %(getTime()-startofCalc);
  
  while(norm > 1.e-5 && iter < 200) {      
    J.setFvec(params, residue);

    params -= 0.01*residue;
    residual.updateOrbitals(params);

    //residue *=-1.;
    //VectorXd x = gmres.solve(residue);
    //params += 0.1*x;
    //residual.updateOrbitals(params);

    iter++;
    
    func(params, residue);
    norm = residue.norm();
    
    std::cout << format("%5i   %14.8f   %14.6f %6.6f \n") %(iter) %(residual.E0)
        %(residue.norm()) %(getTime()-startofCalc);
  }
}
