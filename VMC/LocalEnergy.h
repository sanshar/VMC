#ifndef LOCALENERGY_HEADER_H
#define LOCALENERGY_HEADER_H
#include <Eigen/Dense>
#include "Determinants.h"
#include "global.h"

template<typename T>
void BuildJastrowVars(const Eigen::Matrix<T, Eigen::Dynamic, 1> &vars, const Determinant &D, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &J, Eigen::Matrix<T, Eigen::Dynamic, 1> &Jmid, int &numVars);

template<typename T>
void BuildOrbitalVars(const Eigen::Matrix<T, Eigen::Dynamic, 1> &vars, const Determinant &D, std::array<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>, 2> &R, int &numVars);

template<typename T>
T JastrowSlaterLocalEnergy(const Determinant &D, const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &J, const Eigen::Matrix<T, Eigen::Dynamic, 1> &Jmid, const std::array<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>, 2> &R);

template <typename T>
void BuildPfaffianVars(const Eigen::Matrix<T, Eigen::Dynamic, 1> &vars, const Determinant &D, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &pairMat, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &thetaInv, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &fMat, int &numVars);

template<typename T>
T JastrowPfaffianLocalEnergy(const Determinant &D, const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &J, const Eigen::Matrix<T, Eigen::Dynamic, 1> &Jmid, const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &pairMat, const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> thetaInv, const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> fMat);

class LocalEnergySolver
{
    public:
    Determinant D;

    LocalEnergySolver(const Determinant &_D) : D(_D) {}

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
          std::array<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>, 2> R;
          BuildOrbitalVars<T> (vars, D, R, numVars);

          //Local energy evaluation
          return JastrowSlaterLocalEnergy<T> (D, J, Jmid, R);
        }
        else if (schd.wavefunctionType == "JastrowPfaffian")
        {
          int numVars = 0;
      
          //Jastrow vars
          Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> J;
          Eigen::Matrix<T, Eigen::Dynamic, 1> Jmid;
          BuildJastrowVars<T> (vars, D, J, Jmid, numVars);
      
          //Pfaffian vars
          Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> pairMat, thetaInv, fMat;
          BuildPfaffianVars<T> (vars, D, pairMat, thetaInv, fMat, numVars); 
      
          //Local energy evaluation
          return JastrowPfaffianLocalEnergy<T> (D, J, Jmid, pairMat, thetaInv, fMat); 
        }
    }
};
#endif
