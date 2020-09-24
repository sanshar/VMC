# DICE-QMC

This repository contains various state-of-the-art quantum Monte Carlo algorithms implemented and developed in Sandeep Sharma's research group at the University of Colorado at Boulder. Unlike many *ab-initio* methods that scale exponentially with system size, Monte Carlo methods have created a wide breath of highly accurate methods that scale polynomially with system size, albeit with a large prefactor. These methods include [variational Monte Carlo (VMC)](https://en.wikipedia.org/wiki/Variational_Monte_Carlo), [diffusion Monte Carlo (DMC)](https://en.wikipedia.org/wiki/Diffusion_Monte_Carlo), Green's function Monte Carlo (GFMC), Full Configuration Interaction quantum Monte Carlo (FCIQMC), and [Auxilery field quantum Monte Carlo (AFQMC)](https://en.wikipedia.org/wiki/Auxiliary-field_Monte_Carlo), just to name a few. Many of these methods are implemented or in the process of being implemented as a part of this software package.

This repositry was developed with the use of [PySCF](https://github.com/sunqm/pyscf/blob/master/README.md) and we recommend it's use as the starting point for any calculation.

Prerequisites
-------------

To compile this package, one requires:

* [Boost](http://www.boost.org/) This is a set of libraries for the C++ programming language that provides support for tasks and structures such as linear algebra, pseudorandom number generation, multithreading, image processing, regular expressions, and unit testing. When compiling the Boost library make sure that you use the same compiler as you do for *Dice*.

An example of download and compilation commands for the `NN` version of Boost can be:

```
  wget https://dl.bintray.com/boostorg/release/1.64.0/source/boost_1_NN_0.tar.gz
  tar -xf boost_1_NN_0.tar.gz
  cd boost_1_NN_0
  ./bootstrap.sh
  echo "using mpi ;" >> project-config.jam
  ./b2 -j6 --target=shared,static
```

* [Eigen](http://eigen.tuxfamily.org/dox/) This is a high-level C++ library of template headers for linear algebra, matrix and vector operations, geometrical transformations, numerical solvers and related algorithms. Eigen consists of header files and does not have to be compiled but can be installed.

One way of downloading and installing the Eigen package with cmake is:

```
  hg clone https://bitbucket.org/eigen/eigen/
  cd eigen
  mkdir build_dir
  cd build_dir
  cmake ..
  sudo make install
```

* [libigl](https://github.com/libigl/libigl) This is a simple C++ geometry processing library available on github. It is header only and does not need to be installed.

* [Stan](https://github.com/stan-dev/math) This is a C++ math template library for automatic differentiation of any order using forward, reverse, and mixed modes. This requires a number of dependinces that are clearly outlined on their github repository along with easy to follow directions to install all of them. Note it requires Eigen and Boost, two dependencies we have already discussed.

* [Taco](https://github.com/tensor-compiler/taco) The Tensor Algebra Compiler (taco) computes sparse tensor expressions on CPUs and GPUs. To install, there are simple and clear instructions on their github page.

* [PySCF](https://github.com/sunqm/pyscf/blob/master/README.md) This is a python based quantum chemistry software package. We recommend its use with this package and if one is interested in real space calculations this is also a dependency. Installation is straigtforward with the instructions on their github page.

* About compiler requirements:
    - GNU: g++ 7 or newer
    - Intel: icpc 17.0.1 or newer
    - In any case: the C++14 standards must be supported.

Compilation
-------

Edit the `Makefile` in the main directory and change the paths to point to your dependencies.
The user can choose whether to use gcc or intel by setting the `USE_INTEL` variable accordingly,
and whether or not to compile with MPI by setting the `USE_MPI` variable. 
We heavily recommend the intel compilers and MPI, as any challenging calculation will require speed and parallelization.
All the lines in the `Makefile` that normally need to be edited are shown below:

```
 USE_MPI = yes
 USE_INTEL = yes
 
 EIGEN=/projects/ilsa8974/apps/eigen/
 BOOST=/projects/ilsa8974/apps/boost_1_66_0/
 LIBIGL=/projects/ilsa8974/apps/libigl/include/
 PYSCF=/projects/ilsa8974/apps/pyscf/pyscf/lib/
 LIBCINT=/projects/ilsa8974/apps/pyscf/pyscf/lib/deps/lib
 SUNDIALS=/projects/ilsa8974/apps/sundials-3.1.0/stage/include
 STAN=/projects/ilsa8974/apps/math
 TBB=/curc/sw/intel/17.4/compilers_and_libraries_2017.4.196/linux/tbb/
 TACO=/projects/ilsa8974/apps/taco/install

```
