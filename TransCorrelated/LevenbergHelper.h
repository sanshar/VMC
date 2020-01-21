#pragma once

template<typename _Scalar, int NX = Eigen::Dynamic, int NY = Eigen::Dynamic>
struct Functor
{

  // Information that tells the caller the numeric type (eg. double) and size (input / output dim)
  typedef _Scalar Scalar;
  enum { // Required by numerical differentiation module
    InputsAtCompileTime = NX,
          ValuesAtCompileTime = NY
  };

  // Tell the caller the matrix sizes associated with the input, output, and jacobian
  typedef Eigen::Matrix<Scalar,InputsAtCompileTime,1> InputType;
  typedef Eigen::Matrix<Scalar,ValuesAtCompileTime,1> ValueType;
  typedef Eigen::Matrix<Scalar,ValuesAtCompileTime,InputsAtCompileTime> JacobianType;

  // Local copy of the number of inputs
  int m_inputs, m_values;

  // Two constructors:
  Functor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
  Functor(int inputs, int values) : m_inputs(inputs), m_values(values) {}

  // Get methods for users to determine function input and output dimensions
  int inputs() const { return m_inputs; }
  int values() const { return m_values; }

};

template<typename GradFunctor>
struct totalGradWrapper : Functor<double>{
  GradFunctor& ptotalGrad;
  double Value;
  
  totalGradWrapper(GradFunctor& totalGrad, int neq, int nvars)
      : Functor<double>(neq, nvars), ptotalGrad(totalGrad) {};
  
  int operator()(const VectorXd& a, VectorXd& b) const {
    double E = ptotalGrad(a, b);;
    *const_cast<double*>(&Value) = E;
    return 0;
  };
};



