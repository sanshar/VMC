#include "optimizer.h"
#include "Residuals.h"
#include <iostream>
#include "DirectJacobian.h"
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>
#include "global.h"
#include "diis.h"
#include <boost/format.hpp>
using namespace boost;
using namespace std;




void optimizeJastrowParams(
    VectorXd& params,
    boost::function<double (const VectorXd&, VectorXd&)>& func) {

  
  cout << "optimize jastrow "<<endl;
  double Energy, norm=10.;
  int iter = 0;
  VectorXd residue(params.size());

  DIIS diis(10, params.size());
  
  DirectJacobian<double> J(func);
  Eigen::GMRES<DirectJacobian<double>, Eigen::IdentityPreconditioner> gmres;
  gmres.compute(J);
  gmres.setTolerance(1.e-3);
  Energy = func(params, residue);
  norm = residue.norm();
  
  std::cout << format("%5i   %14.8f   %14.6f %6.6f \n") %(iter) %(Energy)
      %(residue.norm()) %(getTime()-startofCalc);

  while(norm > 1.e-5) {      
    J.setFvec(params, residue);

    residue *=-1.;
    VectorXd x = gmres.solve(residue);
    params += x;

    diis.update(params, residue);
    iter++;

    Energy = func(params, residue);
    norm = residue.norm();
    
    std::cout << format("%5i   %14.8f   %14.6f %6.6f \n") %(iter) %(Energy)
        %(residue.norm()) %(getTime()-startofCalc);
  }
}


void optimizeOrbitalParams(
    VectorXd& params,
    boost::function<double (const VectorXd&, VectorXd&)>& func) {

  double Energy, norm=10.;
  int iter = 0;
  VectorXd residue(params.size());

  DIIS diis(8, params.size());
  

  Energy = func(params, residue);
  norm = residue.norm();

  std::cout << format("%5i   %14.8f   %14.6f %6.6f \n") %(iter) %(Energy)
      %(norm) %(getTime()-startofCalc);
  
  while(norm > 1.e-5 && iter < 5000) {      

    //residue *= 0.1;
    params -=  0.01*residue;
    //diis.update(params, residue);

    //residual.updateOrbitals(params);

    iter++;
    
    Energy = func(params, residue);
    norm = residue.norm();
    
    std::cout << format("%5i   %.10f   %.6f %6.6f \n") %(iter) %(Energy)
        %(norm) %(getTime()-startofCalc);
  }
}
