#ifndef LOCALENERGY_HEADER_H
#define LOCALENERGY_HEADER_H
#include <Eigen/Dense>
#include "Determinants.h"
#include "global.h"
#include "Complex.h"

template<typename T>
void BuildJastrowVars(const Eigen::Matrix<T, Eigen::Dynamic, 1> &vars, const Determinant &D, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &J, Eigen::Matrix<T, Eigen::Dynamic, 1> &Jmid, int &numVars);

void concatenateGhf(const vector<int>& v1, const vector<int>& v2, vector<int>& result);

template<typename T>
void BuildOrbitalVars(const Eigen::Matrix<T, Eigen::Dynamic, 1> &vars, const Determinant &D, std::array<Complex<T>, 2> &thetaDet, std::array<Eigen::Matrix<Complex<T>, Eigen::Dynamic, Eigen::Dynamic>, 2> &R, int &numVars);

template<typename T>
T JastrowSlaterLocalEnergy(const Determinant &D, const workingArray &work, const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &J, const Eigen::Matrix<T, Eigen::Dynamic, 1> &Jmid, const std::array<Complex<T>, 2> &thetaDet, const std::array<Eigen::Matrix<Complex<T>, Eigen::Dynamic, Eigen::Dynamic>, 2> &R);

template <typename T>
void BuildPfaffianVars(const Eigen::Matrix<T, Eigen::Dynamic, 1> &vars, const Determinant &D, Eigen::Matrix<Complex<T>, Eigen::Dynamic, Eigen::Dynamic> &pairMat, Eigen::Matrix<Complex<T>, Eigen::Dynamic, Eigen::Dynamic> &thetaInv, Complex<T> &thetaPfaff, std::array<Eigen::Matrix<Complex<T>, Eigen::Dynamic, Eigen::Dynamic>, 2> &rTable, int &numVars);

template<typename T>
T JastrowPfaffianLocalEnergy(const Determinant &D, const workingArray &work, const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &J, const Eigen::Matrix<T, Eigen::Dynamic, 1> &Jmid, const Eigen::Matrix<Complex<T>, Eigen::Dynamic, Eigen::Dynamic> &pairMat, const Eigen::Matrix<Complex<T>, Eigen::Dynamic, Eigen::Dynamic> &thetaInv, const Complex<T> &thetaPfaff, const std::array<Eigen::Matrix<Complex<T>, Eigen::Dynamic, Eigen::Dynamic>, 2> &rTable);

class LocalEnergySolver
{
    public:
    Determinant D;
    workingArray &work;

    LocalEnergySolver(const Determinant &_D, workingArray &_work) : D(_D), work(_work) {}

    template <typename T>
    T operator()(const Eigen::Matrix<T, Eigen::Dynamic, 1> &vars) const
    {
        if (schd.wavefunctionType == "jastrowslater")
        {
          int numVars = 0;

          //Jastrow vars
          Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> J;
          Eigen::Matrix<T, Eigen::Dynamic, 1> Jmid;
          BuildJastrowVars<T> (vars, D, J, Jmid, numVars);

          //CI vars
          std::vector<T> ciExpansion;
          for (int i = 0; i < 1; i++)
          {
            ciExpansion.push_back(vars[numVars + i]);
          }
          numVars += ciExpansion.size();

          //orbital vars
          std::array<Eigen::Matrix<Complex<T>, Eigen::Dynamic, Eigen::Dynamic>, 2> R;
          std::array<Complex<T>, 2> thetaDet;
          BuildOrbitalVars<T> (vars, D, thetaDet, R, numVars);

          //Local energy evaluation
          return JastrowSlaterLocalEnergy<T> (D, work, J, Jmid, thetaDet, R);
        }
        else if (schd.wavefunctionType == "jastrowpfaffian")
        {
          int numVars = 0;
      
          //Jastrow vars
          Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> J;
          Eigen::Matrix<T, Eigen::Dynamic, 1> Jmid;
          BuildJastrowVars<T> (vars, D, J, Jmid, numVars);
      
          //Pfaffian vars
          Eigen::Matrix<Complex<T>, Eigen::Dynamic, Eigen::Dynamic> pairMat, thetaInv;
          Complex<T> thetaPfaff;
          std::array<Eigen::Matrix<Complex<T>, Eigen::Dynamic, Eigen::Dynamic>, 2> rTable;
          BuildPfaffianVars<T> (vars, D, pairMat, thetaInv, thetaPfaff, rTable, numVars); 
      
          //Local energy evaluation
          return JastrowPfaffianLocalEnergy<T> (D, work, J, Jmid, pairMat, thetaInv, thetaPfaff, rTable); 
        }
    }
};

//this function is just used in the LocalEnergySolverJastrow function
void getHfRelIndicesNew(int i, int &relI, int a, int &relA, bool sz, const std::array<std::vector<int>, 2> &closedOrbs, const std::array<std::vector<int>, 2> &openOrbs);

class LocalEnergySolverJastrow
{
    public:
    Determinant D;
    workingArray &work;
    const std::array<std::complex<double>, 2> &thetaDet;
    const std::array<Eigen::MatrixXcd, 2> &R;

    LocalEnergySolverJastrow(const Determinant &_D, workingArray &_work, const std::array<std::complex<double>, 2> &_thetaDet, const std::array<Eigen::MatrixXcd, 2> &_R) : D(_D), work(_work), thetaDet(_thetaDet), R(_R) {}

    template <typename T>
    T operator()(const Eigen::Matrix<T, Eigen::Dynamic, 1> &vars) const
    {
        if (schd.wavefunctionType == "jastrowslater")
        {
          int numVars = 0;

          //Jastrow vars
          Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> J;
          Eigen::Matrix<T, Eigen::Dynamic, 1> Jmid;
          BuildJastrowVars<T> (vars, D, J, Jmid, numVars);

          //Local energy evaluation
          int norbs = Determinant::norbs;
          int nalpha = Determinant::nalpha;
          int nbeta = Determinant::nbeta;
          std::array<vector<int>, 2> closedOrbs, openOrbs;
          D.getOpenClosedAlphaBeta(openOrbs[0], closedOrbs[0], openOrbs[1], closedOrbs[1]);
          T Eloc = D.Energy(I1, I2, coreE);
          for (int l = 0; l < work.nExcitations; l++)
          {
            int ex1 = work.excitation1[l], ex2 = work.excitation2[l];
            double tia = work.HijElement[l];

            int i = ex1 / 2 / norbs, a = ex1 - 2 * norbs * i;
            int j = ex2 / 2 / norbs, b = ex2 - 2 * norbs * j;

            int ri, ra, rj, rb;
            int sz1, sz2;

            T ovlpRatio = 1.0;
            if (j == b && b == 0) //single excitations
            {
                ovlpRatio *= Jmid[a] / Jmid[i] / J(std::max(i,a), std::min(i,a));
                if (i % 2 == 0) //alpha
                    sz1 = 0; 
                else    //beta
                    sz1 = 1;
                getHfRelIndicesNew(i / 2, ri, a / 2, ra, sz1, closedOrbs, openOrbs);
                ovlpRatio *= (R[sz1](ra, ri) * thetaDet[0] * thetaDet[1]).real() / (thetaDet[0] * thetaDet[1]).real();
            }
            else //double excitations
            {
                ovlpRatio *= Jmid[a] * Jmid[b] * J(std::max(a,b), std::min(a,b)) * J(std::max(i,j), std::min(i,j)) / Jmid[i] / Jmid[j] / J(std::max(a,i), std::min(a,i)) / J(std::max(a,j), std::min(a,j)) / J(std::max(i,b), std::min(i,b)) / J(std::max(j,b), std::min(j,b));
                if (i % 2 == j % 2 && i % 2 == 0) //aa to aa
                {
                    sz1 = 0;
                    sz2 = 0;
                }
                else if (i % 2 == j % 2 && i % 2 == 1) //bb to bb
                {
                    sz1 = 1;
                    sz2 = 1;
                }
                else if (i % 2 != j % 2 && i % 2 == 0) //ab to ab
                {
                    sz1 = 0;
                    sz2 = 1;
                }
                else  //ba to ba
                {
                    sz1 = 1;
                    sz2 = 0;
                }
                getHfRelIndicesNew(i / 2, ri, a / 2, ra, sz1, closedOrbs, openOrbs);
                getHfRelIndicesNew(j / 2, rj, b / 2, rb, sz2, closedOrbs, openOrbs); 
                if (sz1 == sz2 || schd.hf == "ghf")
                {
                    ovlpRatio *= ((R[sz1](ra, ri) * R[sz1](rb, rj) - R[sz1](rb, ri) * R[sz1](ra, rj))* thetaDet[0] * thetaDet[1]).real() / (thetaDet[0] * thetaDet[1]).real();
                }
                else
                {
                    ovlpRatio *= ((R[sz1](ra, ri) * R[sz2](rb, rj)) * thetaDet[0] * thetaDet[1]).real() / (thetaDet[0] * thetaDet[1]).real();
                }
            }
            Eloc += tia * ovlpRatio;
          }
          return Eloc;
        }
        else
        {
          cout << "This is not implemented for any other wavefunction" << endl;
          return 0.0;
        }
    }
};

void getHfRelIndices(int i, int &relI, int a, int &relA, bool sz, const std::array<std::vector<int>, 2> &closedOrbs, const std::array<std::vector<int>, 2> &openOrbs);

class LocalEnergySolverOrbital
{
    public:
    Determinant D;
    workingArray &work;
    Eigen::MatrixXd J;
    Eigen::VectorXd Jmid;

    LocalEnergySolverOrbital(const Determinant &_D, workingArray &_work, const Eigen::VectorXd &corrVars) : D(_D), work(_work)
    {
        int dummy = 0;
        BuildJastrowVars<double>(corrVars, D, J, Jmid, dummy);
    }

    template <typename T>
    T operator()(const Eigen::Matrix<T, Eigen::Dynamic, 1> &vars) const
    {
        if (schd.wavefunctionType == "jastrowslater")
        {
          int numVars = 0;

          //CI vars
          std::vector<T> ciExpansion;
          for (int i = 0; i < 1; i++)
          {
            ciExpansion.push_back(vars[numVars + i]);
          }
          numVars += ciExpansion.size();

          //orbital vars
          std::array<Eigen::Matrix<Complex<T>, Eigen::Dynamic, Eigen::Dynamic>, 2> R;
          std::array<Complex<T>, 2> thetaDet;
          BuildOrbitalVars<T> (vars, D, thetaDet, R, numVars);

          //Local energy evaluation
          int norbs = Determinant::norbs;
          int nalpha = Determinant::nalpha;
          int nbeta = Determinant::nbeta;
          std::array<vector<int>, 2> closedOrbs, openOrbs;
          D.getOpenClosedAlphaBeta(openOrbs[0], closedOrbs[0], openOrbs[1], closedOrbs[1]);
          T Eloc = D.Energy(I1, I2, coreE);
          for (int l = 0; l < work.nExcitations; l++)
          {
            int ex1 = work.excitation1[l], ex2 = work.excitation2[l];
            double tia = work.HijElement[l];

            int i = ex1 / 2 / norbs, a = ex1 - 2 * norbs * i;
            int j = ex2 / 2 / norbs, b = ex2 - 2 * norbs * j;

            int ri, ra, rj, rb;
            int sz1, sz2;

            T ovlpRatio = 1.0;
            if (j == b && b == 0) //single excitations
            {
                ovlpRatio *= Jmid[a] / Jmid[i] / J(std::max(i,a), std::min(i,a));
                if (i % 2 == 0) //alpha
                    sz1 = 0; 
                else    //beta
                    sz1 = 1;
                getHfRelIndices(i / 2, ri, a / 2, ra, sz1, closedOrbs, openOrbs);
                ovlpRatio *= (R[sz1](ra, ri) * thetaDet[0] * thetaDet[1]).real() / (thetaDet[0] * thetaDet[1]).real();
            }
            else //double excitations
            {
                ovlpRatio *= Jmid[a] * Jmid[b] * J(std::max(a,b), std::min(a,b)) * J(std::max(i,j), std::min(i,j)) / Jmid[i] / Jmid[j] / J(std::max(a,i), std::min(a,i)) / J(std::max(a,j), std::min(a,j)) / J(std::max(i,b), std::min(i,b)) / J(std::max(j,b), std::min(j,b));
                if (i % 2 == j % 2 && i % 2 == 0) //aa to aa
                {
                    sz1 = 0;
                    sz2 = 0;
                }
                else if (i % 2 == j % 2 && i % 2 == 1) //bb to bb
                {
                    sz1 = 1;
                    sz2 = 1;
                }
                else if (i % 2 != j % 2 && i % 2 == 0) //ab to ab
                {
                    sz1 = 0;
                    sz2 = 1;
                }
                else  //ba to ba
                {
                    sz1 = 1;
                    sz2 = 0;
                }
                getHfRelIndices(i / 2, ri, a / 2, ra, sz1, closedOrbs, openOrbs);
                getHfRelIndices(j / 2, rj, b / 2, rb, sz2, closedOrbs, openOrbs); 
                if (sz1 == sz2 || schd.hf == "ghf")
                {
                    ovlpRatio *= ((R[sz1](ra, ri) * R[sz1](rb, rj) - R[sz1](rb, ri) * R[sz1](ra, rj))* thetaDet[0] * thetaDet[1]).real() / (thetaDet[0] * thetaDet[1]).real();
                }
                else
                {
                    ovlpRatio *= ((R[sz1](ra, ri) * R[sz2](rb, rj)) * thetaDet[0] * thetaDet[1]).real() / (thetaDet[0] * thetaDet[1]).real();
                }
            }
            Eloc += tia * ovlpRatio;
          }
          return Eloc;
        }
        else
        {
          cout << "This is not implemented for any other wavefunction" << endl;
          return 0.0;
        }
    }
};
#endif
