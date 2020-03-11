#ifndef JASTROWBASIS_HEADER_H
#define JASTROWBASIS_HEADER_H
#include "input.h"
#include "global.h"
#include <fstream>
#include <vector>
#include <Eigen/Dense>
//#include <boost/serialization/serialization.hpp>
using namespace std;
using namespace Eigen;

//Number Counter
  void NC_eval(int elec, const vector<Vector3d> &x, VectorXd &values)
  {
    vector<double>& Ncharge = schd.Ncharge;
    vector<int>& Nbasis = schd.Nbasis;
    vector<Vector3d>& Ncoords = schd.Ncoords;

    int norbs = schd.basis->getNorbs();
    vector<double> aoValues(10*norbs, 0.0);
    schd.basis->eval(x[elec], &aoValues[0]);

    VectorXd num = VectorXd::Zero(Ncharge.size());
    double denom = 0.0;
    int orb = 0;
    for (int I = 0; I < num.size(); I++)
    {
        for (int mu = 0; mu < Nbasis[I]; mu++)
        {
            num[I] += aoValues[orb] * aoValues[orb];
            denom += aoValues[orb] * aoValues[orb];
            orb++;
        }
    }
    values = num / denom;
  }


  void NC_eval_deriv(int elec, const vector<Vector3d> &x, VectorXd &values, array<VectorXd, 3> &grad)
  {
    vector<double>& Ncharge = schd.Ncharge;
    vector<int>& Nbasis = schd.Nbasis;
    vector<Vector3d>& Ncoords = schd.Ncoords;

    int norbs = schd.basis->getNorbs();
    vector<double> aoValues(10*norbs, 0.0);
    schd.basis->eval_deriv2(x[elec], &aoValues[0]);

    VectorXd num = VectorXd::Zero(Ncharge.size());
    VectorXd gradxNum = VectorXd::Zero(Ncharge.size());
    VectorXd gradyNum = VectorXd::Zero(Ncharge.size());
    VectorXd gradzNum = VectorXd::Zero(Ncharge.size());
    double denom = 0.0;
    Vector3d gradDenom = Vector3d::Zero(3);
    int orb = 0;
    for (int I = 0; I < num.size(); I++)
    {
        for (int mu = 0; mu < Nbasis[I]; mu++)
        {
            num[I] += aoValues[orb] * aoValues[orb];
            denom += aoValues[orb] * aoValues[orb];
            
            //gradx
            gradxNum[I] += 2.0 * aoValues[orb] * aoValues[1*norbs + orb];
            gradDenom[0] += 2.0 * aoValues[orb] * aoValues[1*norbs + orb];
            //grady
            gradyNum[I] += 2.0 * aoValues[orb] * aoValues[2*norbs + orb];
            gradDenom[1] += 2.0 * aoValues[orb] * aoValues[2*norbs + orb];
            //gradz
            gradzNum[I] += 2.0 * aoValues[orb] * aoValues[3*norbs + orb];
            gradDenom[2] += 2.0 * aoValues[orb] * aoValues[3*norbs + orb];
            orb++;
        }
    }
    values = num / denom;
    grad[0] = (gradxNum * denom - num * gradDenom[0]) / (denom * denom);
    grad[1] = (gradyNum * denom - num * gradDenom[1]) / (denom * denom);
    grad[2] = (gradzNum * denom - num * gradDenom[2]) / (denom * denom);
  }


  void NC_eval_deriv2(int elec, const vector<Vector3d> &x, VectorXd &values, array<VectorXd, 3> &grad, array<VectorXd, 3> &grad2)
  {
    vector<double>& Ncharge = schd.Ncharge;
    vector<int>& Nbasis = schd.Nbasis;
    vector<Vector3d>& Ncoords = schd.Ncoords;

    int norbs = schd.basis->getNorbs();
    vector<double> aoValues(10*norbs, 0.0);
    schd.basis->eval_deriv2(x[elec], &aoValues[0]);

    VectorXd num = VectorXd::Zero(Ncharge.size());
    VectorXd gradxNum = VectorXd::Zero(Ncharge.size());
    VectorXd gradyNum = VectorXd::Zero(Ncharge.size());
    VectorXd gradzNum = VectorXd::Zero(Ncharge.size());
    VectorXd gradxxNum = VectorXd::Zero(Ncharge.size());
    VectorXd gradyyNum = VectorXd::Zero(Ncharge.size());
    VectorXd gradzzNum = VectorXd::Zero(Ncharge.size());
    double denom = 0.0;
    Vector3d gradDenom = Vector3d::Zero(3);
    Vector3d grad2Denom = Vector3d::Zero(3);
    int orb = 0;
    for (int I = 0; I < num.size(); I++)
    {
        for (int mu = 0; mu < Nbasis[I]; mu++)
        {
            num[I] += aoValues[orb] * aoValues[orb];
            denom += aoValues[orb] * aoValues[orb];
            
            //gradx
            gradxNum[I] += 2.0 * aoValues[orb] * aoValues[1*norbs + orb];
            gradDenom[0] += 2.0 * aoValues[orb] * aoValues[1*norbs + orb];
            gradxxNum[I] += 2.0 * aoValues[1*norbs + orb] * aoValues[1*norbs + orb] + 2.0 * aoValues[orb] * aoValues[4*norbs + orb];
            grad2Denom[0] += 2.0 * aoValues[1*norbs + orb] * aoValues[1*norbs + orb] + 2.0 * aoValues[orb] * aoValues[4*norbs + orb];

            //grady
            gradyNum[I] += 2.0 * aoValues[orb] * aoValues[2*norbs + orb];
            gradDenom[1] += 2.0 * aoValues[orb] * aoValues[2*norbs + orb];
            gradyyNum[I] += 2.0 * aoValues[2*norbs + orb] * aoValues[2*norbs + orb] + 2.0 * aoValues[orb] * aoValues[7*norbs + orb];
            grad2Denom[1] += 2.0 * aoValues[2*norbs + orb] * aoValues[2*norbs + orb] + 2.0 * aoValues[orb] * aoValues[7*norbs + orb];

            //gradz
            gradzNum[I] += 2.0 * aoValues[orb] * aoValues[3*norbs + orb];
            gradDenom[2] += 2.0 * aoValues[orb] * aoValues[3*norbs + orb];
            gradzzNum[I] += 2.0 * aoValues[3*norbs + orb] * aoValues[3*norbs + orb] + 2.0 * aoValues[orb] * aoValues[9*norbs + orb];
            grad2Denom[2] += 2.0 * aoValues[3*norbs + orb] * aoValues[3*norbs + orb] + 2.0 * aoValues[orb] * aoValues[9*norbs + orb];

            orb++;
        }
    }
    values = num / denom;
    grad[0] = (gradxNum * denom - num * gradDenom[0]) / (denom * denom);
    grad[1] = (gradyNum * denom - num * gradDenom[1]) / (denom * denom);
    grad[2] = (gradzNum * denom - num * gradDenom[2]) / (denom * denom);

    grad2[0] = (denom * denom * gradxxNum
               - denom * (2.0 * gradxNum * gradDenom[0] + num * grad2Denom[0])
               + 2.0 * num * gradDenom[0] * gradDenom[0]) / (denom * denom * denom);
    grad2[1] = (denom * denom * gradyyNum
               - denom * (2.0 * gradyNum * gradDenom[1] + num * grad2Denom[1])
               + 2.0 * num * gradDenom[1] * gradDenom[1]) / (denom * denom * denom);
    grad2[2] = (denom * denom * gradzzNum
               - denom * (2.0 * gradzNum * gradDenom[2] + num * grad2Denom[2])
               + 2.0 * num * gradDenom[2] * gradDenom[2]) / (denom * denom * denom);
  }



//Iliya style, square of orbital value
  void AB_eval(int elec, const vector<Vector3d> &x, VectorXd &values)
  {
    int norbs = schd.basis->getNorbs();

    vector<double> aoValues(10*norbs, 0.0);
    schd.basis->eval(x[elec], &aoValues[0]);

    VectorXd num = VectorXd::Zero(norbs);
    double denom = 0.0;
    for (int I = 0; I < norbs; I++)
    {
        num[I] += aoValues[I] * aoValues[I];
        denom += aoValues[I] * aoValues[I];
    }

    values = num / denom;
  }


  void AB_eval_deriv(int elec, const vector<Vector3d> &x, VectorXd &values, array<VectorXd, 3> &grad)
  {
    int norbs = schd.basis->getNorbs();

    vector<double> aoValues(10*norbs, 0.0);
    schd.basis->eval_deriv2(x[elec], &aoValues[0]);

    VectorXd num = VectorXd::Zero(norbs);
    VectorXd gradxNum = VectorXd::Zero(norbs);
    VectorXd gradyNum = VectorXd::Zero(norbs);
    VectorXd gradzNum = VectorXd::Zero(norbs);
    double denom = 0.0;
    Vector3d gradDenom = Vector3d::Zero(3);
    for (int I = 0; I < norbs; I++)
    {
        num[I] += aoValues[I] * aoValues[I];
        denom += aoValues[I] * aoValues[I];
        
        //gradx
        gradxNum[I] += 2.0 * aoValues[I] * aoValues[1*norbs + I];
        gradDenom[0] += 2.0 * aoValues[I] * aoValues[1*norbs + I];
        //grady
        gradyNum[I] += 2.0 * aoValues[I] * aoValues[2*norbs + I];
        gradDenom[1] += 2.0 * aoValues[I] * aoValues[2*norbs + I];
        //gradz
        gradzNum[I] += 2.0 * aoValues[I] * aoValues[3*norbs + I];
        gradDenom[2] += 2.0 * aoValues[I] * aoValues[3*norbs + I];
    }
    values = num / denom;
    grad[0] = (gradxNum * denom - num * gradDenom[0]) / (denom * denom);
    grad[1] = (gradyNum * denom - num * gradDenom[1]) / (denom * denom);
    grad[2] = (gradzNum * denom - num * gradDenom[2]) / (denom * denom);
  }


  void AB_eval_deriv2(int elec, const vector<Vector3d> &x, VectorXd &values, array<VectorXd, 3> &grad, array<VectorXd, 3> &grad2)
  {
    int norbs = schd.basis->getNorbs();
    vector<double> aoValues(10*norbs, 0.0);
    schd.basis->eval_deriv2(x[elec], &aoValues[0]);

    VectorXd num = VectorXd::Zero(norbs);
    VectorXd gradxNum = VectorXd::Zero(norbs);
    VectorXd gradyNum = VectorXd::Zero(norbs);
    VectorXd gradzNum = VectorXd::Zero(norbs);
    VectorXd gradxxNum = VectorXd::Zero(norbs);
    VectorXd gradyyNum = VectorXd::Zero(norbs);
    VectorXd gradzzNum = VectorXd::Zero(norbs);
    double denom = 0.0;
    Vector3d gradDenom = Vector3d::Zero(3);
    Vector3d grad2Denom = Vector3d::Zero(3);
    for (int I = 0; I < norbs; I++)
    {
        num[I] += aoValues[I] * aoValues[I];
        denom += aoValues[I] * aoValues[I];
        
        //gradx
        gradxNum[I] += 2.0 * aoValues[I] * aoValues[1*norbs + I];
        gradDenom[0] += 2.0 * aoValues[I] * aoValues[1*norbs + I];
        gradxxNum[I] += 2.0 * aoValues[1*norbs + I] * aoValues[1*norbs + I] + 2.0 * aoValues[I] * aoValues[4*norbs + I];
        grad2Denom[0] += 2.0 * aoValues[1*norbs + I] * aoValues[1*norbs + I] + 2.0 * aoValues[I] * aoValues[4*norbs + I];

        //grady
        gradyNum[I] += 2.0 * aoValues[I] * aoValues[2*norbs + I];
        gradDenom[1] += 2.0 * aoValues[I] * aoValues[2*norbs + I];
        gradyyNum[I] += 2.0 * aoValues[2*norbs + I] * aoValues[2*norbs + I] + 2.0 * aoValues[I] * aoValues[7*norbs + I];
        grad2Denom[1] += 2.0 * aoValues[2*norbs + I] * aoValues[2*norbs + I] + 2.0 * aoValues[I] * aoValues[7*norbs + I];

        //gradz
        gradzNum[I] += 2.0 * aoValues[I] * aoValues[3*norbs + I];
        gradDenom[2] += 2.0 * aoValues[I] * aoValues[3*norbs + I];
        gradzzNum[I] += 2.0 * aoValues[3*norbs + I] * aoValues[3*norbs + I] + 2.0 * aoValues[I] * aoValues[9*norbs + I];
        grad2Denom[2] += 2.0 * aoValues[3*norbs + I] * aoValues[3*norbs + I] + 2.0 * aoValues[I] * aoValues[9*norbs + I];
    }
    values = num / denom;
    grad[0] = (gradxNum * denom - num * gradDenom[0]) / (denom * denom);
    grad[1] = (gradyNum * denom - num * gradDenom[1]) / (denom * denom);
    grad[2] = (gradzNum * denom - num * gradDenom[2]) / (denom * denom);

    grad2[0] = (denom * denom * gradxxNum
               - denom * (2.0 * gradxNum * gradDenom[0] + num * grad2Denom[0])
               + 2.0 * num * gradDenom[0] * gradDenom[0]) / (denom * denom * denom);
    grad2[1] = (denom * denom * gradyyNum
               - denom * (2.0 * gradyNum * gradDenom[1] + num * grad2Denom[1])
               + 2.0 * num * gradDenom[1] * gradDenom[1]) / (denom * denom * denom);
    grad2[2] = (denom * denom * gradzzNum
               - denom * (2.0 * gradzNum * gradDenom[2] + num * grad2Denom[2])
               + 2.0 * num * gradDenom[2] * gradDenom[2]) / (denom * denom * denom);
  }
#endif
