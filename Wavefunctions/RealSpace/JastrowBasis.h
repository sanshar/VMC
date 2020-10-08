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


//s-orbital Number Counter
  void sNC_eval(int elec, const vector<Vector3d> &x, VectorXd &values) //note that s orbital do not need to be squared
  {
    vector<double>& Ncharge = schd.Ncharge;
    vector<int>& Nbasis = schd.Nbasis;
    vector<vector<int>>& NSbasis = schd.NSbasis;
    vector<Vector3d>& Ncoords = schd.Ncoords;

    int norbs = schd.basis->getNorbs();
    vector<double> aoValues(10*norbs, 0.0);
    schd.basis->eval(x[elec], &aoValues[0]);

    VectorXd num = VectorXd::Zero(Ncharge.size());
    double denom = 0.0;
    for (int I = 0; I < NSbasis.size(); I++)
    {
        for (int mu = 0; mu < NSbasis[I].size(); mu++)
        {
            int orb = NSbasis[I][mu];
            num[I] += aoValues[orb];
            denom += aoValues[orb];
        }
    }
    values = num / denom;
  }


  void sNC_eval_deriv(int elec, const vector<Vector3d> &x, VectorXd &values, array<VectorXd, 3> &grad)
  {
    vector<double>& Ncharge = schd.Ncharge;
    vector<int>& Nbasis = schd.Nbasis;
    vector<vector<int>>& NSbasis = schd.NSbasis;
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
    for (int I = 0; I < NSbasis.size(); I++)
    {
        for (int mu = 0; mu < NSbasis[I].size(); mu++)
        {
            int orb = NSbasis[I][mu];

            num[I] += aoValues[orb];
            denom += aoValues[orb];
            
            //gradx
            gradxNum[I] += aoValues[1*norbs + orb];
            gradDenom[0] += aoValues[1*norbs + orb];
            //grady
            gradyNum[I] += aoValues[2*norbs + orb];
            gradDenom[1] += aoValues[2*norbs + orb];
            //gradz
            gradzNum[I] += aoValues[3*norbs + orb];
            gradDenom[2] += aoValues[3*norbs + orb];
        }
    }
    values = num / denom;
    grad[0] = (gradxNum * denom - num * gradDenom[0]) / (denom * denom);
    grad[1] = (gradyNum * denom - num * gradDenom[1]) / (denom * denom);
    grad[2] = (gradzNum * denom - num * gradDenom[2]) / (denom * denom);
  }


  void sNC_eval_deriv2(int elec, const vector<Vector3d> &x, VectorXd &values, array<VectorXd, 3> &grad, array<VectorXd, 3> &grad2)
  {
    vector<double>& Ncharge = schd.Ncharge;
    vector<int>& Nbasis = schd.Nbasis;
    vector<vector<int>>& NSbasis = schd.NSbasis;
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
    for (int I = 0; I < NSbasis.size(); I++)
    {
        for (int mu = 0; mu < NSbasis[I].size(); mu++)
        {
            int orb = NSbasis[I][mu];

            num[I] += aoValues[orb];
            denom += aoValues[orb];
            
            //gradx
            gradxNum[I] += aoValues[1*norbs + orb];
            gradDenom[0] += aoValues[1*norbs + orb];

            gradxxNum[I] += aoValues[4*norbs + orb];
            grad2Denom[0] += aoValues[4*norbs + orb];

            //grady
            gradyNum[I] += aoValues[2*norbs + orb];
            gradDenom[1] += aoValues[2*norbs + orb];

            gradyyNum[I] += aoValues[7*norbs + orb];
            grad2Denom[1] += aoValues[7*norbs + orb];

            //gradz
            gradzNum[I] += aoValues[3*norbs + orb];
            gradDenom[2] += aoValues[3*norbs + orb];

            gradzzNum[I] += aoValues[9*norbs + orb];
            grad2Denom[2] += aoValues[9*norbs + orb];
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
  void AB2_eval(int elec, const vector<Vector3d> &x, VectorXd &values)
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


  void AB2_eval_deriv(int elec, const vector<Vector3d> &x, VectorXd &values, array<VectorXd, 3> &grad)
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


  void AB2_eval_deriv2(int elec, const vector<Vector3d> &x, VectorXd &values, array<VectorXd, 3> &grad, array<VectorXd, 3> &grad2)
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


//s-orbital AB2 style
  void sAB2_eval(int elec, const vector<Vector3d> &x, VectorXd &values) //note that s orbital do not need to be squared
  {
    vector<double>& Ncharge = schd.Ncharge;
    vector<int>& Nbasis = schd.Nbasis;
    vector<vector<int>>& NSbasis = schd.NSbasis;
    vector<Vector3d>& Ncoords = schd.Ncoords;

    int norbs = schd.basis->getNorbs();
    vector<double> aoValues(10*norbs, 0.0);
    schd.basis->eval(x[elec], &aoValues[0]);

    int numSorb = 0;
    for (int i = 0; i < NSbasis.size(); i++) { numSorb += NSbasis[i].size(); }

    VectorXd num = VectorXd::Zero(numSorb);
    double denom = 0.0;
    int index = 0;
    for (int I = 0; I < NSbasis.size(); I++)
    {
        for (int mu = 0; mu < NSbasis[I].size(); mu++)
        {
            int orb = NSbasis[I][mu];

            num[index] += aoValues[orb] * aoValues[orb];
            denom += aoValues[orb] * aoValues[orb];

            index++;
        }
    }
    values = num / denom;
  }


  void sAB2_eval_deriv(int elec, const vector<Vector3d> &x, VectorXd &values, array<VectorXd, 3> &grad)
  {
    vector<double>& Ncharge = schd.Ncharge;
    vector<int>& Nbasis = schd.Nbasis;
    vector<vector<int>>& NSbasis = schd.NSbasis;
    vector<Vector3d>& Ncoords = schd.Ncoords;

    int norbs = schd.basis->getNorbs();
    vector<double> aoValues(10*norbs, 0.0);
    schd.basis->eval_deriv2(x[elec], &aoValues[0]);

    int numSorb = 0;
    for (int i = 0; i < NSbasis.size(); i++) { numSorb += NSbasis[i].size(); }

    VectorXd num = VectorXd::Zero(numSorb);
    VectorXd gradxNum = VectorXd::Zero(numSorb);
    VectorXd gradyNum = VectorXd::Zero(numSorb);
    VectorXd gradzNum = VectorXd::Zero(numSorb);
    double denom = 0.0;
    Vector3d gradDenom = Vector3d::Zero(3);
    int index = 0;
    for (int I = 0; I < NSbasis.size(); I++)
    {
        for (int mu = 0; mu < NSbasis[I].size(); mu++)
        {
            int orb = NSbasis[I][mu];

            num[index] += aoValues[orb] * aoValues[orb];
            denom += aoValues[orb] * aoValues[orb];
            
            //gradx
            gradxNum[index] += 2.0 * aoValues[orb] * aoValues[1*norbs + orb];
            gradDenom[0] += 2.0 * aoValues[orb] * aoValues[1*norbs + orb];
            //grady
            gradyNum[index] += 2.0 * aoValues[orb] * aoValues[2*norbs + orb];
            gradDenom[1] += 2.0 * aoValues[orb] * aoValues[2*norbs + orb];
            //gradz
            gradzNum[index] += 2.0 * aoValues[orb] * aoValues[3*norbs + orb];
            gradDenom[2] += 2.0 * aoValues[orb] * aoValues[3*norbs + orb];

            index++;
        }
    }
    values = num / denom;
    grad[0] = (gradxNum * denom - num * gradDenom[0]) / (denom * denom);
    grad[1] = (gradyNum * denom - num * gradDenom[1]) / (denom * denom);
    grad[2] = (gradzNum * denom - num * gradDenom[2]) / (denom * denom);
  }


  void sAB2_eval_deriv2(int elec, const vector<Vector3d> &x, VectorXd &values, array<VectorXd, 3> &grad, array<VectorXd, 3> &grad2)
  {
    vector<double>& Ncharge = schd.Ncharge;
    vector<int>& Nbasis = schd.Nbasis;
    vector<vector<int>>& NSbasis = schd.NSbasis;
    vector<Vector3d>& Ncoords = schd.Ncoords;

    int norbs = schd.basis->getNorbs();
    vector<double> aoValues(10*norbs, 0.0);
    schd.basis->eval_deriv2(x[elec], &aoValues[0]);

    int numSorb = 0;
    for (int i = 0; i < NSbasis.size(); i++) { numSorb += NSbasis[i].size(); }

    VectorXd num = VectorXd::Zero(numSorb);
    VectorXd gradxNum = VectorXd::Zero(numSorb);
    VectorXd gradyNum = VectorXd::Zero(numSorb);
    VectorXd gradzNum = VectorXd::Zero(numSorb);
    VectorXd gradxxNum = VectorXd::Zero(numSorb);
    VectorXd gradyyNum = VectorXd::Zero(numSorb);
    VectorXd gradzzNum = VectorXd::Zero(numSorb);
    double denom = 0.0;
    Vector3d gradDenom = Vector3d::Zero(3);
    Vector3d grad2Denom = Vector3d::Zero(3);
    int index = 0;
    for (int I = 0; I < NSbasis.size(); I++)
    {
        for (int mu = 0; mu < NSbasis[I].size(); mu++)
        {
            int orb = NSbasis[I][mu];

            num[index] += aoValues[orb] * aoValues[orb];
            denom += aoValues[orb] * aoValues[orb];
            
            //gradx
            gradxNum[index] += 2.0 * aoValues[orb] * aoValues[1*norbs + orb];
            gradDenom[0] += 2.0 * aoValues[orb] * aoValues[1*norbs + orb];

            gradxxNum[index] += 2.0 * aoValues[1*norbs + orb] * aoValues[1*norbs + orb] + 2.0 * aoValues[orb] * aoValues[4*norbs + orb];
            grad2Denom[0] += 2.0 * aoValues[1*norbs + orb] * aoValues[1*norbs + orb] + 2.0 * aoValues[orb] * aoValues[4*norbs + orb];

            //grady
            gradyNum[index] += 2.0 * aoValues[orb] * aoValues[2*norbs + orb];
            gradDenom[1] += 2.0 * aoValues[orb] * aoValues[2*norbs + orb];

            gradyyNum[index] += 2.0 * aoValues[2*norbs + orb] * aoValues[2*norbs + orb] + 2.0 * aoValues[orb] * aoValues[7*norbs + orb];
            grad2Denom[1] += 2.0 * aoValues[2*norbs + orb] * aoValues[2*norbs + orb] + 2.0 * aoValues[orb] * aoValues[7*norbs + orb];

            //gradz
            gradzNum[index] +=  2.0 * aoValues[orb] * aoValues[3*norbs + orb];
            gradDenom[2] += 2.0 * aoValues[orb] * aoValues[3*norbs + orb];

            gradzzNum[index] += 2.0 * aoValues[3*norbs + orb] * aoValues[3*norbs + orb] + 2.0 * aoValues[orb] * aoValues[9*norbs + orb];
            grad2Denom[2] += 2.0 * aoValues[3*norbs + orb] * aoValues[3*norbs + orb] + 2.0 * aoValues[orb] * aoValues[9*norbs + orb];

            index++;
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


//Sorella style, orbital value
  void SS_eval(int elec, const vector<Vector3d> &x, VectorXd &values)
  {
    int norbs = schd.basis->getNorbs();

    vector<double> aoValues(10*norbs, 0.0);
    schd.basis->eval(x[elec], &aoValues[0]);

    values = VectorXd::Zero(norbs);
    for (int I = 0; I < norbs; I++)
    {
        values[I] = aoValues[I];
    }
  }


  void SS_eval_deriv(int elec, const vector<Vector3d> &x, VectorXd &values, array<VectorXd, 3> &grad)
  {
    int norbs = schd.basis->getNorbs();

    vector<double> aoValues(10*norbs, 0.0);
    schd.basis->eval_deriv2(x[elec], &aoValues[0]);

    values = VectorXd::Zero(norbs);
    grad[0] = VectorXd::Zero(norbs);
    grad[1] = VectorXd::Zero(norbs);
    grad[2] = VectorXd::Zero(norbs);
    for (int I = 0; I < norbs; I++)
    {
      values[I] = aoValues[I];
      grad[0][I] = aoValues[1*norbs + I];
      grad[1][I] = aoValues[2*norbs + I];
      grad[2][I] = aoValues[3*norbs + I];
    }
  }


  void SS_eval_deriv2(int elec, const vector<Vector3d> &x, VectorXd &values, array<VectorXd, 3> &grad, array<VectorXd, 3> &grad2)
  {
    int norbs = schd.basis->getNorbs();

    vector<double> aoValues(10*norbs, 0.0);
    schd.basis->eval_deriv2(x[elec], &aoValues[0]);

    values = VectorXd::Zero(norbs);
    grad[0] = VectorXd::Zero(norbs);
    grad[1] = VectorXd::Zero(norbs);
    grad[2] = VectorXd::Zero(norbs);
    grad2[0] = VectorXd::Zero(norbs);
    grad2[1] = VectorXd::Zero(norbs);
    grad2[2] = VectorXd::Zero(norbs);
    for (int I = 0; I < norbs; I++)
    {
      values[I] = aoValues[I];
      grad[0][I] = aoValues[1*norbs + I];
      grad[1][I] = aoValues[2*norbs + I];
      grad[2][I] = aoValues[3*norbs + I];
      grad2[0][I] = aoValues[4*norbs + I];
      grad2[1][I] = aoValues[7*norbs + I];
      grad2[2][I] = aoValues[9*norbs + I];
    }
  }


//Single Gaussian with inputed variance

  void SG_eval(int elec, const vector<Vector3d> &x, VectorXd &values)
  {
    const VectorXd &relec = x[elec];
    int natm = schd.Ncoords.size();
    double N = 1.0 / (schd.sigma * std::sqrt(2 * M_PI));
    double var = schd.sigma * schd.sigma;

    VectorXd num = VectorXd::Zero(natm);
    double denom = 0.0;
    for (int I = 0; I < natm; I++)
    {
      VectorXd r = relec - schd.Ncoords[I];
      double r2 = r.squaredNorm();
      double arg = - (1.0 / 2.0) * r2 / var;
      double val = N * std::exp(arg);

      num[I] += val;
      denom += val;
    }
    
    values = num / denom; 
  }

  void SG_eval_deriv(int elec, const vector<Vector3d> &x, VectorXd &values, array<VectorXd, 3> &grad)
  {
    const VectorXd &relec = x[elec];
    int natm = schd.Ncoords.size();
    double N = 1.0 / (schd.sigma * std::sqrt(2 * M_PI));
    double var = schd.sigma * schd.sigma;

    VectorXd num = VectorXd::Zero(natm);
    VectorXd gradxNum = VectorXd::Zero(natm);
    VectorXd gradyNum = VectorXd::Zero(natm);
    VectorXd gradzNum = VectorXd::Zero(natm);
    Vector3d gradDenom = Vector3d::Zero(3);
    double denom = 0.0;
    for (int I = 0; I < natm; I++)
    {
      VectorXd r = relec - schd.Ncoords[I];
      double r2 = r.squaredNorm();
      double arg = - (1.0 / 2.0) * r2 / var;
      double val = N * std::exp(arg);

      num[I] += val;
      denom += val;

      //gradx
      gradxNum[I] += - r(0) * val / var;
      gradDenom[0] += - r(0) * val / var;
      //grady
      gradyNum[I] += - r(1) * val / var;
      gradDenom[1] += - r(1) * val / var;
      //gradz
      gradzNum[I] += - r(2) * val / var;
      gradDenom[2] += - r(2) * val / var;
    } 

    values = num / denom; 
    grad[0] = (gradxNum * denom - num * gradDenom[0]) / (denom * denom);
    grad[1] = (gradyNum * denom - num * gradDenom[1]) / (denom * denom);
    grad[2] = (gradzNum * denom - num * gradDenom[2]) / (denom * denom);
  }

  void SG_eval_deriv2(int elec, const vector<Vector3d> &x, VectorXd &values, array<VectorXd, 3> &grad, array<VectorXd, 3> &grad2)
  {
    const VectorXd &relec = x[elec];
    int natm = schd.Ncoords.size();
    double N = 1.0 / (schd.sigma * std::sqrt(2 * M_PI));
    double var = schd.sigma * schd.sigma;

    VectorXd num = VectorXd::Zero(natm);
    VectorXd gradxNum = VectorXd::Zero(natm);
    VectorXd gradyNum = VectorXd::Zero(natm);
    VectorXd gradzNum = VectorXd::Zero(natm);
    VectorXd gradxxNum = VectorXd::Zero(natm);
    VectorXd gradyyNum = VectorXd::Zero(natm);
    VectorXd gradzzNum = VectorXd::Zero(natm);
    double denom = 0.0;
    Vector3d gradDenom = Vector3d::Zero(3);
    Vector3d grad2Denom = Vector3d::Zero(3);
    for (int I = 0; I < natm; I++)
    {
      VectorXd r = relec - schd.Ncoords[I];
      double r2 = r.squaredNorm();
      double arg = - (1.0 / 2.0) * r2 / var;
      double val = N * std::exp(arg);

      num[I] += val;
      denom += val;

      //gradx
      gradxNum[I] += - r(0) * val / var;
      gradDenom[0] += - r(0) * val / var;
      gradxxNum[I] += - (var - r(0) * r(0)) * val / (var * var);
      grad2Denom[0] += - (var - r(0) * r(0)) * val / (var * var);

      //grady
      gradyNum[I] += - r(1) * val / var;
      gradDenom[1] += - r(1) * val / var;
      gradyyNum[I] += - (var - r(1) * r(1)) * val / (var * var);
      grad2Denom[1] += - (var - r(1) * r(1)) * val / (var * var);

      //gradz
      gradzNum[I] += - r(2) * val / var;
      gradDenom[2] += - r(2) * val / var;
      gradzzNum[I] += - (var - r(2) * r(2)) * val / (var * var);
      grad2Denom[2] += - (var - r(2) * r(2)) * val / (var * var);
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


//Grid of Gaussians, read from file
  //Gaussian function, derivatives with respect to ri
  double Gaussian(const Vector3d &ri, const Vector3d &rI, double sigma, Vector3d &grad, Vector3d &grad2)
  {
    double variance = sigma * sigma;
    double N = std::pow(M_PI * variance, 3.0 / 4.0);

    VectorXd r = ri - rI;
    double r2 = r.squaredNorm();
    double arg = - (1.0 / 2.0) * (r2 / variance);

    double val = std::exp(arg) / N;

    grad[0] = - r(0) * val / variance;
    grad2[0] = - (variance - r(0) * r(0)) * val / (variance * variance);

    grad[1] = - r(1) * val / variance;
    grad2[1] = - (variance - r(1) * r(1)) * val / (variance * variance);

    grad[2] = - r(2) * val / variance;
    grad2[2] = - (variance - r(2) * r(2)) * val / (variance * variance);

    return val;
  }

  void G_eval(int elec, const vector<Vector3d> &x, VectorXd &values)
  {
    const VectorXd &relec = x[elec];
    int ngrid = schd.gridGaussians.size();

    VectorXd num = VectorXd::Zero(ngrid);
    double denom = 0.0;
    for (int I = 0; I < ngrid; I++)
    {
      double sigma = schd.gridGaussians[I].first;
      Vector3d &rI = schd.gridGaussians[I].second;
      Vector3d g, g2;
      double val = Gaussian(relec, rI, sigma, g, g2);

      num[I] += val;
      denom += val;
    }
    
    values = num / denom; 
  }

  void G_eval_deriv(int elec, const vector<Vector3d> &x, VectorXd &values, array<VectorXd, 3> &grad)
  {
    const VectorXd &relec = x[elec];
    int ngrid = schd.gridGaussians.size();

    VectorXd num = VectorXd::Zero(ngrid);
    VectorXd gradxNum = VectorXd::Zero(ngrid);
    VectorXd gradyNum = VectorXd::Zero(ngrid);
    VectorXd gradzNum = VectorXd::Zero(ngrid);
    Vector3d gradDenom = Vector3d::Zero(3);
    double denom = 0.0;
    for (int I = 0; I < ngrid; I++)
    {
      double sigma = schd.gridGaussians[I].first;
      Vector3d &rI = schd.gridGaussians[I].second;
      Vector3d g, g2;
      double val = Gaussian(relec, rI, sigma, g, g2);

      num[I] += val;
      denom += val;

      //gradx
      gradxNum[I] += g(0);
      gradDenom[0] += g(0);
      //grady
      gradyNum[I] += g(1);
      gradDenom[1] += g(1);
      //gradz
      gradzNum[I] += g(2);
      gradDenom[2] += g(2);
    } 

    values = num / denom; 
    grad[0] = (gradxNum * denom - num * gradDenom[0]) / (denom * denom);
    grad[1] = (gradyNum * denom - num * gradDenom[1]) / (denom * denom);
    grad[2] = (gradzNum * denom - num * gradDenom[2]) / (denom * denom);
  }

  void G_eval_deriv2(int elec, const vector<Vector3d> &x, VectorXd &values, array<VectorXd, 3> &grad, array<VectorXd, 3> &grad2)
  {
    const VectorXd &relec = x[elec];
    int ngrid = schd.gridGaussians.size();

    VectorXd num = VectorXd::Zero(ngrid);
    VectorXd gradxNum = VectorXd::Zero(ngrid);
    VectorXd gradyNum = VectorXd::Zero(ngrid);
    VectorXd gradzNum = VectorXd::Zero(ngrid);
    VectorXd gradxxNum = VectorXd::Zero(ngrid);
    VectorXd gradyyNum = VectorXd::Zero(ngrid);
    VectorXd gradzzNum = VectorXd::Zero(ngrid);
    double denom = 0.0;
    Vector3d gradDenom = Vector3d::Zero(3);
    Vector3d grad2Denom = Vector3d::Zero(3);
    for (int I = 0; I < ngrid; I++)
    {
      double sigma = schd.gridGaussians[I].first;
      Vector3d &rI = schd.gridGaussians[I].second;
      Vector3d g, g2;
      double val = Gaussian(relec, rI, sigma, g, g2);

      num[I] += val;
      denom += val;

      //gradx
      gradxNum[I] += g(0);
      gradDenom[0] += g(0);
      gradxxNum[I] += g2(0);
      grad2Denom[0] += g2(0);
      //grady
      gradyNum[I] += g(1);
      gradDenom[1] += g(1);
      gradyyNum[I] += g2(1);
      grad2Denom[1] += g2(1);
      //gradz
      gradzNum[I] += g(2);
      gradDenom[2] += g(2);
      gradzzNum[I] += g2(2);
      grad2Denom[2] += g2(2);

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
