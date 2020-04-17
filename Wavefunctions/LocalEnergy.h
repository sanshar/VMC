#ifndef LOCALENERGY_HEADER_H
#define LOCALENERGY_HEADER_H
#include <Eigen/Dense>
#include "Determinants.h"
#include "global.h"
#include "Complex.h"

template<typename T>
void BuildJastrowVars(const Eigen::Matrix<T, Eigen::Dynamic, 1> &vars, const Determinant &D, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &J, Eigen::Matrix<T, Eigen::Dynamic, 1> &Jmid, int &numVars);

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
        if (schd.wavefunctionType == "JastrowSlater")
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
        else if (schd.wavefunctionType == "JastrowPfaffian")
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

void getHfRelIndices(int i, int &relI, int a, int &relA, bool sz, const std::array<std::vector<int>, 2> &closedOrbs, const std::array<std::vector<int>, 2> &openOrbs);
#endif
