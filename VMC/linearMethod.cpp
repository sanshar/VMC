#include "linearMethod.h"
#include <Eigen/Dense>
#include <boost/serialization/serialization.hpp>
#include "iowrapper.h"
#include "global.h"
#include "input.h"
#include <iterator>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#include <fstream>

#ifndef SERIAL
#include "mpi.h"
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif

void ConjGrad(const DirectLM &A, const Eigen::VectorXd &u, double theta, const Eigen::VectorXd &b, int n, Eigen::VectorXd &x)
{
  double tol = 1.e-10;

  VectorXd Ap = VectorXd::Zero(x.rows());
  A.multiplyGJD(x, theta, u, Ap);
  VectorXd r = b - Ap;
  VectorXd p = r;
  
  double rsold = r.adjoint() * r;
  if (fabs(rsold) < tol) return;
  
  for (int i = 0; i < n; i++)
  {
    A.multiplyGJD(p, theta, u, Ap);
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

void generalizedJacobiDavidson(const Eigen::MatrixXd &H, const Eigen::MatrixXd &S, double &lambda, Eigen::VectorXd &v)
{
    int dim = H.rows(); //dimension of problem
    int restart = std::max((int) (0.1 * dim), 30); //20 - 30
    int q = std::max((int) (0.04 * dim), 5); //5 - 10
    Eigen::MatrixXd V, HV, SV;  //matrices storing action of vector on sample space
    //Eigen::VectorXd z = Eigen::VectorXd::Random(dim);
    //Eigen::VectorXd z = Eigen::VectorXd::Unit(dim, 0);
    Eigen::VectorXd z = Eigen::VectorXd::Constant(dim, 0.01);
    while (1)
    {
        int m = V.cols(); //number of vectors in subspace at current iteration

        //modified grahm schmidt to orthogonalize sample space with respect to overlap matrix
        Eigen::VectorXd Sz = S * z;
        for (int i = 0; i < m; i++)
        {
            double alpha = V.col(i).adjoint() * Sz;
            z = z - alpha * V.col(i);
        }

        //normalize z after orthogonalization and calculate action of matrices on z
        Sz = S * z;
        double beta = std::sqrt(z.adjoint() * Sz);
        Eigen::VectorXd v_m = z / beta;
        Eigen::VectorXd Hv_m = H * v_m;
        Eigen::VectorXd Sv_m = S * v_m;
        //store new vector
        V.conservativeResize(H.rows(), m + 1);
        HV.conservativeResize(H.rows(), m + 1);
        SV.conservativeResize(H.rows(), m + 1);
        V.col(m) = v_m;
        HV.col(m) = Hv_m;
        SV.col(m) = Sv_m;

        //solve eigenproblem in subspace
        Eigen::MatrixXd A = V.adjoint() * HV;
        Eigen::MatrixXd B = V.adjoint() * SV;
        Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> es(A, B);
        //if sample space size is too large, restart subspace
        if (m == restart)
        {
            Eigen::MatrixXd N = es.eigenvectors().block(0, 0, A.rows(), q);
            V = V * N;
            HV = HV * N;
            SV = SV * N;
            A = V.adjoint() * HV;
            B = V.adjoint() * SV;
            es.compute(A, B);
        }
        Eigen::VectorXd s = es.eigenvectors().col(0);
        double theta = es.eigenvalues()(0);
  cout << "theta: " << theta << endl;

        //transform vector into original space
        Eigen::VectorXd u = V * s;
        //calculate residue vector
        Eigen::VectorXd u_H = HV * s;
        Eigen::VectorXd u_S = SV * s;
        Eigen::VectorXd r = u_H - theta * u_S;

        if (r.squaredNorm() < 1.e-8)
        {
            lambda = theta;
            v = u;
            break;
        }
        
        Eigen::MatrixXd X = Eigen::MatrixXd::Identity(u.rows(), u.rows());
        Eigen::MatrixXd Xleft = X - S * u * u.adjoint();
        Eigen::MatrixXd Xright = X - u * u.adjoint() * S;
        X = Xleft * (H - theta * S) * Xright;
        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> dec(X);
        z = dec.solve(-r);
    }
}

int FindBound(const VectorXd &v, double target)
{
    if (v.size() == 1 || v(0) > target)
    {
      return 0;
    }
    for (int i = 0; i < v.rows() - 1; i++)
    {
      if (v(i) <= target && target < v(i + 1))
      {
        return i;
      }
    }
}

void AppendVectorToSubspace(const DirectLM &H, const Eigen::VectorXd &z, Eigen::MatrixXd &V, Eigen::MatrixXd &HV, Eigen::MatrixXd &SV)
{
    int dim = H.G[0].rows(); //dimension of problem
    int m = V.cols(); 
    Eigen::VectorXd zcopy(z);
    Eigen::VectorXd Sz;
    H.multiplyS(zcopy, Sz);
    for (int i = 0; i < m; i++)
    {
      double alpha = V.col(i).dot(Sz);
      zcopy = zcopy - alpha * V.col(i);
    }
    H.multiplyS(zcopy, Sz);
    double beta = zcopy.dot(Sz);
    if (beta < 0.0)
      beta *= -1.0;
    beta = std::sqrt(beta);
    Eigen::VectorXd v_m = zcopy / (beta + 1.e-8);

    Eigen::VectorXd Hv_m;
    H.multiplyH(v_m, Hv_m);
    Eigen::VectorXd Sv_m;
    H.multiplyS(v_m, Sv_m);

    V.conservativeResize(dim, m + 1);
    HV.conservativeResize(dim, m + 1);
    SV.conservativeResize(dim, m + 1);
    V.col(m) = v_m;
    HV.col(m) = Hv_m;
    SV.col(m) = Sv_m;
}


void SelfAdjointGeneralizedJacobiDavidson(DirectLM &H, double target, const Eigen::VectorXd &targetv, double &lambda, Eigen::VectorXd &v, int n)
{
  int dim = H.G[0].rows(); //dimension of problem
  int restart = std::max((int) (0.05 * dim), 20); //20 - 30
  int q = 5; //number of vectors to restart subspace
  //int q = std::max((int) (0.04 * dim), 5); //5 - 10
  
  Eigen::MatrixXd V, HV, SV;  //matrices storing action of vector on sample space

  Eigen::VectorXd z = targetv;
  AppendVectorToSubspace(H, z, V, HV, SV);


/*
if (commrank == 0)
{
    cout << endl << "iter: " << iter << endl;
    cout <<"target: " << tau << endl;
    cout <<"r2: " <<  old_r2 << endl;
}
*/
  int iter = 0;
  double tau = target;
  double old_r2 = 1.0;
  double old_theta = 1.0;
  while(1)
  {
    iter++;
    Eigen::MatrixXd A = V.adjoint() * HV;
    Eigen::MatrixXd B = V.adjoint() * SV;
    Eigen::GeneralizedSelfAdjointEigenSolver<MatrixXd> es(A, B);
    int index = FindBound(es.eigenvalues(), tau);

    if (V.cols() == restart)
    {
      Eigen::MatrixXd N = es.eigenvectors().block(0, index, A.rows(), q);
      V = V * N;
      HV = HV * N;
      SV = SV * N;
      A = V.adjoint() * HV;
      B = V.adjoint() * SV;
      es.compute(A, B);
      index = FindBound(es.eigenvalues(), tau);
    }

    double theta = es.eigenvalues()(index);
    Eigen::VectorXd s = es.eigenvectors().col(index);
    Eigen::VectorXd u = V * s;
    Eigen::VectorXd u_H = HV * s;
    Eigen::VectorXd u_S = SV * s;
    Eigen::VectorXd r = u_H - theta * u_S;
    double r2 = r.squaredNorm();

if (commrank == 0)
{
    cout << endl << "iter: " << iter << endl;
    cout << "subspace problem" << endl << endl;
    cout << "theta: " << tau << endl;
    cout << "target: " << tau << endl;
    cout << "r2: " <<  r2 << endl;
    cout << "thetadiff " << std::abs(theta - old_theta) << endl;
    cout << endl << es.eigenvalues().transpose() << endl << endl;
}

    if (r2 < old_r2)
    {
      tau = theta;
      old_r2 = r2;
    } 
    if ((r2 < 1.e-8) || (std::abs(theta - old_theta) < 1.e-8) || std::isnan(theta))
    {
      lambda = theta;
      v = u;
      return;
    }
    old_theta = theta;
    z = Eigen::VectorXd::Unit(dim, 0);
    ConjGrad(H, u, theta, -r, n, z);
    AppendVectorToSubspace(H, z, V, HV, SV);
  }
}                                       

void CanonicalTransform(const Eigen::MatrixXd &Smatrix, Eigen::MatrixXd &X)
{
  X.setZero(Smatrix.rows(), Smatrix.cols());
  Eigen::SelfAdjointEigenSolver<MatrixXd> oes(Smatrix);
  int index = 0;
  for (int i = 0; i < X.cols(); i++)
  {
    double eigval = oes.eigenvalues()(i);
    if (abs(eigval) > 1.e-10)
    {
      X.col(index) = oes.eigenvectors().col(i) / sqrt(eigval);
      index++;
    }
  }
  X.conservativeResize(Smatrix.rows(), index);
} 

void SortEig(Eigen::VectorXd &D, Eigen::MatrixXd &V)
{
  Eigen::VectorXd Dcopy(D);
  Eigen::MatrixXd Vcopy(V);
  std::vector<double> a(Dcopy.data(), Dcopy.data() + Dcopy.size());
  // initialize original index locations
  std::vector<int> idx(a.size());
  std::iota(idx.begin(), idx.end(), 0);
  // sort indexes based on comparing values in a
  std::sort(idx.begin(), idx.end(), [&a](int i1, int i2) -> bool {return a[i1] < a[i2];});
  for (int i = 0; i < idx.size(); i++)
  {
    D(i) = Dcopy(idx[i]);
    V.col(i) = Vcopy.col(idx[i]);
  }   
}

void GeneralizedJacobiDavidson(DirectLM &H, double target, const Eigen::VectorXd &targetv, double &lambda, Eigen::VectorXd &v, int cgIter, double tol)
{
  int dim = H.G[0].rows(); //dimension of problem
  //int restart = std::max((int) (0.05 * dim), 30); //20 - 30
  int restart = 30;
  int q = 5; //number of vectors to restart subspace
  //int q = std::max((int) (0.04 * dim), 5); //5 - 10
  
  Eigen::MatrixXd V, HV, SV;  //matrices storing action of vector on sample space
  Eigen::VectorXd z = targetv;
  AppendVectorToSubspace(H, z, V, HV, SV);

  double tau = target;
  double old_rNorm = 1.0;
  int iter = 0;
  while(1)
  {
    //solve subspace problem
    iter++;
    Eigen::MatrixXd A = V.adjoint() * HV;
    Eigen::MatrixXd B = V.adjoint() * SV;
    Eigen::GeneralizedEigenSolver<MatrixXd> es(A, B);
    Eigen::VectorXd D = es.eigenvalues().real();
    Eigen::MatrixXd U = es.eigenvectors().real();
    SortEig(D, U);
    int index = FindBound(D, tau);
    double theta = D(index);
    Eigen::VectorXd s = U.col(index);
    Eigen::VectorXd u = V * s;
    Eigen::VectorXd u_H = HV * s;
    Eigen::VectorXd u_S = SV * s;
    Eigen::VectorXd r = u_H - theta * u_S;
    double rNorm = r.norm();
    //if eigensolver fails
    if (std::isnan(rNorm) || std::isinf(rNorm))
    {
      if (commrank == 0 && schd.printOpt) cout << "EIGENSOLVER FAIL" << endl;
      Eigen::MatrixXd X;
      CanonicalTransform(B, X);
      Eigen::MatrixXd A_prime = X.adjoint() * A * X;
      Eigen::EigenSolver<MatrixXd> es(A_prime);
      Eigen::VectorXd D = es.eigenvalues().real();
      Eigen::MatrixXd U = X * es.eigenvectors().real();
      SortEig(D, U);
      index = FindBound(D, tau);
      theta = D(index);
      s = U.col(index);
      u = V * s;
      u_H = HV * s;
      u_S = SV * s;
      r = u_H - theta * u_S;
      rNorm = r.norm();
    }
    //solve for correction vector
    z = Eigen::VectorXd::Unit(dim, 0);
    ConjGrad(H, u, theta, -r, cgIter, z);

    if (commrank == 0 && schd.printOpt)
    {
      cout << endl << "____________" << iter << "____________" << endl;
      cout << "theta: " << theta << endl;
      cout << "best guess so far: " << tau << endl;
      cout << "residual norm: " <<  rNorm << endl;
      cout << "size of subspace: " << V.cols() << endl;
    }
  
    //update best guess
    if (rNorm < old_rNorm)
    {
      tau = theta;
      old_rNorm = rNorm;
    } 
    //check for convergence
    if (rNorm < tol)
    {
      lambda = theta;
      v = u;
      return;
    }
    //restart subspace
    if (V.cols() >= restart)
    {
      Eigen::MatrixXd N = U.block(0, index, U.rows(), q);
      V = V * N;
      HV = HV * N;
      SV = SV * N;
    }
    AppendVectorToSubspace(H, z, V, HV, SV);
  }
}
