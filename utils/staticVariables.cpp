
#include "Determinants.h"
#include "rDeterminants.h"
#include <boost/interprocess/managed_shared_memory.hpp>
#include "Eigen/Dense"
#include <string>
#include <ctime>
#include <sys/time.h>
#include "input.h"
#include "Profile.h"
#include "integral.h"
#include "readSlater.h"
#ifndef SERIAL
#include "mpi.h"
#endif

int rDeterminant::nelec = 1;
int rDeterminant::nalpha = 1;
int rDeterminant::nbeta = 1;

int Determinant::norbs = 1;
int Determinant::nalpha = 1;
int Determinant::nbeta = 1;
int Determinant::EffDetLen = 1;
char Determinant::Trev = 0;

vector<string> slaterParser::AtomSymbols{"X",\
                              "H" , "He", "Li", "Be", "B" , "C" , "N" , "O" , "F" , "Ne",\
                              "Na", "Mg", "Al", "Si", "P" , "S" , "Cl", "Ar", "K" , "Ca",\
                              "Sc", "Ti", "V" , "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",\
                              "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y" , "Zr",\
                              "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",\
                              "Sb", "Te", "I" , "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",\
                              "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",\
                              "Lu", "Hf", "Ta", "W" , "Re", "Os", "Ir", "Pt", "Au", "Hg",\
                              "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",\
                              "Pa", "U" , "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",\
                              "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",\
                              "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"};

twoInt I2;
oneInt I1;
double coreE;
twoIntHeatBathSHM I2hb(1e-10);

Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic> Determinant::LexicalOrder ;

boost::interprocess::shared_memory_object int2Segment;
boost::interprocess::mapped_region regionInt2;
std::string shciint2;

boost::interprocess::shared_memory_object int2SHMSegment;
boost::interprocess::mapped_region regionInt2SHM;
std::string shciint2shm;

std::mt19937 generator;

#ifndef SERIAL
MPI_Comm shmcomm, localcomm;
#endif
int commrank, shmrank, localrank;
int commsize, shmsize, localsize;

schedule schd;
Profile prof;

double getTime() {
  struct timeval start;
  gettimeofday(&start, NULL);
  return start.tv_sec + 1.e-6*start.tv_usec;
}
double startofCalc;

void license() {
  return;
  if (commrank == 0) {
  cout << endl;
  cout << endl;
  cout << "**************************************************************"<<endl;
  cout << "Dice  Copyright (C) 2017  Sandeep Sharma"<<endl;
  cout <<"This program is distributed in the hope that it will be useful,"<<endl;
  cout <<"but WITHOUT ANY WARRANTY; without even the implied warranty of"<<endl;
  cout <<"MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE."<<endl;
  cout <<"See the GNU General Public License for more details."<<endl;
  cout << endl<<endl;
  cout << "Author:       Sandeep Sharma"<<endl;
  cout << "Please visit our group page for up to date information on other projects"<<endl;
  cout << "http://www.colorado.edu/lab/sharmagroup/"<<endl;
  cout << "**************************************************************"<<endl;
  cout << endl;
  cout << endl;
  }
}


