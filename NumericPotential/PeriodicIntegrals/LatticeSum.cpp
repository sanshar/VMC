#include <algorithm>
#include <numeric> 

#include "LatticeSum.h"
#include "BasisShell.h"

double dotProduct(double* vA, double* vB) {
  return vA[0] * vB[0] + vA[1] * vB[1] + vA[2] * vB[2];
};

void crossProduct(double* v_A, double* v_B, double* c_P, double factor=1.) {
  c_P[0] = factor * (v_A[1] * v_B[2] - v_A[2] * v_B[1]);
  c_P[1] = -factor * (v_A[0] * v_B[2] - v_A[2] * v_B[0]);
  c_P[2] = factor * (v_A[0] * v_B[1] - v_A[1] * v_B[0]);
};

double getKLattice(double* KLattice, double* Lattice) {
  vector<double> cross(3);
  crossProduct(&Lattice[3], &Lattice[6], &cross[0]);
  double Volume = dotProduct(&Lattice[0], &cross[0]);
  
  crossProduct(&Lattice[3], &Lattice[6], &KLattice[0], 2*M_PI/Volume);
  crossProduct(&Lattice[6], &Lattice[0], &KLattice[3], 2*M_PI/Volume);
  crossProduct(&Lattice[0], &Lattice[3], &KLattice[6], 2*M_PI/Volume);
  return Volume;
};
double dist(double a, double b, double c) {
  return a*a + b*b + c*c;
}


void LatticeSum::getRelativeCoords(BasisShell *pA, BasisShell *pC,
                                   double& Tx, double& Ty, double& Tz) {
  double Txmin = pA->Xcoord - pC->Xcoord,
      Tymin = pA->Ycoord - pC->Ycoord,
      Tzmin = pA->Zcoord - pC->Zcoord;
  Tx = Txmin; Ty = Tymin; Tz = Tzmin;

  if (Txmin == 0 && Tymin == 0 && Tzmin == 0) {
    return;
  }

  for (int nx=-1; nx<=1; nx++)
  for (int ny=-1; ny<=1; ny++)
  for (int nz=-1; nz<=1; nz++)
  {
    if (dist(Tx + nx * RLattice[0] + ny * RLattice[3] + nz * RLattice[6],
             Ty + nx * RLattice[1] + ny * RLattice[4] + nz * RLattice[7],
             Tz + nx * RLattice[2] + ny * RLattice[5] + nz * RLattice[8])
        < dist(Txmin, Tymin, Tzmin))  {
      Txmin = Tx + nx * RLattice[0] + ny * RLattice[3] + nz * RLattice[6];
      Tymin = Tx + nx * RLattice[1] + ny * RLattice[4] + nz * RLattice[7];
      Tzmin = Tz + nx * RLattice[2] + ny * RLattice[5] + nz * RLattice[8];
    }
  }

  Tx = Txmin; Ty = Tymin; Tz = Tzmin;
}

LatticeSum::LatticeSum(double* Lattice, int nr, int nk, double _Eta2Rho,
                       double _Eta2RhoCoul, double pscreen) {
  
  //int nr = 1, nk = 10;
  int ir = 0; int Nr = 4*nr+1;
  vector<double> Rcoordcopy(3*Nr*Nr*Nr), Rdistcopy(Nr*Nr*Nr);

  int Nrkeep =  pow(2*nr+1, 3);
  Rcoord.resize(3*Nrkeep);
  Rdist.resize(Nrkeep);

  //rvals
  for (int i = -2*nr; i<=2*nr ; i++)
  for (int j = -2*nr; j<=2*nr ; j++)
  for (int k = -2*nr; k<=2*nr ; k++) {
    Rcoordcopy[3*ir+0] = i * Lattice[0] + j * Lattice[3] + k * Lattice[6];
    Rcoordcopy[3*ir+1] = i * Lattice[1] + j * Lattice[4] + k * Lattice[7];
    Rcoordcopy[3*ir+2] = i * Lattice[2] + j * Lattice[5] + k * Lattice[8];

    Rdistcopy[ir] = Rcoordcopy[3*ir+0] * Rcoordcopy[3*ir+0]
        +  Rcoordcopy[3*ir+1] * Rcoordcopy[3*ir+1]
        +  Rcoordcopy[3*ir+2] * Rcoordcopy[3*ir+2];
    ir++;
  }
  
  //sort rcoord and rdist in ascending order
  std::vector<int> idx(Nr*Nr*Nr);
  std::iota(idx.begin(), idx.end(), 0);
  std::stable_sort(idx.begin(), idx.end(),
                   [&Rdistcopy](size_t i1, size_t i2) {return Rdistcopy[i1] < Rdistcopy[i2];});
  for (int i=0; i<Nrkeep; i++) {
    Rdist[i] = Rdistcopy[idx[i]];
    Rcoord[3*i+0] = Rcoordcopy[3*idx[i]+0];
    Rcoord[3*i+1] = Rcoordcopy[3*idx[i]+1];
    Rcoord[3*i+2] = Rcoordcopy[3*idx[i]+2];
  }

  
  //make klattice
  KLattice.resize(9), RLattice.resize(9);
  for (int i=0; i<9; i++) RLattice[i] = Lattice[i];
  RVolume = getKLattice(&KLattice[0], Lattice);

  int Nk = 2*nk +1;
  vector<double> Kcoordcopy(3*(Nk*Nk*Nk-1)), Kdistcopy(Nk*Nk*Nk-1);
  Kcoord.resize(3 * (Nk*Nk*Nk-1));
  Kdist.resize(Nk*Nk*Nk-1);

  ir = 0;
  //kvals
  for (int i = -nk; i<=nk ; i++)
  for (int j = -nk; j<=nk ; j++)
  for (int k = -nk; k<=nk ; k++) {
    if (i == 0 && j == 0 && k == 0) continue;
    Kcoordcopy[3*ir+0] = i * KLattice[0] + j * KLattice[3] + k * KLattice[6];
    Kcoordcopy[3*ir+1] = i * KLattice[1] + j * KLattice[4] + k * KLattice[7];
    Kcoordcopy[3*ir+2] = i * KLattice[2] + j * KLattice[5] + k * KLattice[8];

    Kdistcopy[ir] = Kcoordcopy[3*ir+0] * Kcoordcopy[3*ir+0]
                 +  Kcoordcopy[3*ir+1] * Kcoordcopy[3*ir+1]
                 +  Kcoordcopy[3*ir+2] * Kcoordcopy[3*ir+2];

    ir++;
  }

  idx.resize(Nk*Nk*Nk-1);
  std::iota(idx.begin(), idx.end(), 0);
  std::stable_sort(idx.begin(), idx.end(),
                   [&Kdistcopy](size_t i1, size_t i2) {return Kdistcopy[i1] < Kdistcopy[i2];});
  
  for (int i=0; i<idx.size(); i++) {
    Kdist[i] = Kdistcopy[idx[i]];
    Kcoord[3*i+0] = Kcoordcopy[3*idx[i]+0];
    Kcoord[3*i+1] = Kcoordcopy[3*idx[i]+1];
    Kcoord[3*i+2] = Kcoordcopy[3*idx[i]+2];
  }

  Eta2RhoOvlp = _Eta2Rho/(Rdist[Nrkeep-1]);
  Eta2RhoCoul = _Eta2RhoCoul/(Rdist[1]);
  screen = pscreen;
  cout << Eta2RhoOvlp<<"  "<<Eta2RhoCoul<<endl;

}


void LatticeSum::printLattice() {
  cout <<"Rlattice: "<<endl;
  for (int i=0; i<3; i++) {
    for (int j=0; j<3; j++)
      printf("%8.4f ", RLattice[j*3+i]);
    cout << endl;
  }

  cout <<"Klattice: "<<endl;
  for (int i=0; i<3; i++) {
    for (int j=0; j<3; j++)
      printf("%8.4f ", KLattice[j*3+i]);
    cout << endl;
  }
  
}

void LatticeSum::getIncreasingIndex(size_t *&idx, double Tx, double Ty, double Tz, ct::FMemoryStack& Mem) {

  double* Tdist;
  Mem.Alloc(idx, Rdist.size());
  Mem.Alloc(Tdist, Rdist.size());
  for (int i=0; i<Rdist.size(); i++) {
    idx[i] = i;
    Tdist[i] = dist(Tx + Rcoord[3*i + 0],
                    Ty + Rcoord[3*i + 1],
                    Tz + Rcoord[3*i + 2]);
  }
  std::stable_sort(idx, (idx+Rdist.size()),
                   [&Tdist](size_t i1, size_t i2) {return Tdist[i1] < Tdist[i2];});
  
  Mem.Free(Tdist);
}
