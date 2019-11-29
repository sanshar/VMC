#include "calcRDM.h"

complex<T> calcRDM::getRDM(int a, int b, MatrixXcT& rdm) {
  return rdm(b, a);
}

complex<T> calcRDM::getRDM(int a, int b, int c, int d, MatrixXcT& rdm){
  rows[0] = d; rows[1] = c;
  cols[0] = a; cols[1] = b;
  Slice(rdm, rows, cols, rdmval2);
  return rdmval2.determinant();
}

complex<T> calcRDM::getRDM(int a, int b, int c, int d, int e, int f, MatrixXcT& rdm){
  rows[0] = f; rows[1] = e; rows[2] = d;
  cols[0] = a; cols[1] = b; cols[2] = c;
  Slice(rdm, rows, cols, rdmval3);
  return rdmval3.determinant();
}

complex<T> calcRDM::getRDM(int a, int b, int c, int d, int e, int f, int g, int h, MatrixXcT& rdm){
  rows[0] = h; rows[1] = g; rows[2] = f; rows[3] = e;
  cols[0] = a; cols[1] = b; cols[2] = c; cols[3] = d; 
  Slice(rdm, rows, cols, rdmval4);
  return rdmval4.determinant();
}

complex<T> calcRDM::getRDM(int a, int b, int c, int d, int e, int f, int g, int h, int i, int j, MatrixXcT& rdm){
  rows[0] = j; rows[1] = i; rows[2] = h; rows[3] = g; rows[4] = f;
  cols[0] = a; cols[1] = b; cols[2] = c; cols[3] = d; cols[4] = e; 
  Slice(rdm, rows, cols, rdmval5);
  return rdmval5.determinant();
}

complex<T> calcRDM::getRDM(int a, int b, int c, int d, int e, int f, int g, int h, int i, int j, int k, int l, MatrixXcT& rdm){
  rows[0] = l; rows[1] = k; rows[2] = j; rows[3] = i; rows[4] = h; rows[5] = g;
  cols[0] = a; cols[1] = b; cols[2] = c; cols[3] = d; cols[4] = e; cols[5] = f;
  Slice(rdm, rows, cols, rdmval6);
  return rdmval6.determinant();
}

//computer generated
complex<T> calcRDM::calcTerm1(int P, int Q, int I, int K, MatrixXcT& rdm) {
  complex<T> contribution = 0;
  if (P == Q && Q == I)
    contribution = 1.0*getRDM(P,K, rdm );

  else if (Q == I)
    contribution = 1.0*getRDM(P,Q,K,P, rdm );

  else if (P == I)
    contribution = -1.0*getRDM(P,Q,K,Q, rdm );

  else if (P == Q)
    contribution = -1.0*getRDM(I,P,K,Q, rdm );

  else
    contribution = -1.0*getRDM(I,P,Q,K,P,Q, rdm );
  return contribution;
}

complex<T> calcRDM::calcTerm1(int P, int Q, int I, int J, int K, int L, MatrixXcT& rdm) {
  if (K == -1 && L == -1)
    return calcTerm1(P, Q, I, J, rdm);
  complex<T> contribution = 0;
  
  if (P == J && Q == I)
    contribution = -1.0*getRDM(P,Q,K,L, rdm );

  else if (P == I && Q == J)
    contribution = 1.0*getRDM(P,Q,K,L, rdm );

  else if (P == Q && Q == J)
    contribution = 1.0*getRDM(I,P,K,L, rdm );

  else if (P == Q && Q == I)
    contribution = -1.0*getRDM(J,P,K,L, rdm );

  else if (Q == J)
    contribution = -1.0*getRDM(I,P,Q,K,L,P, rdm );

  else if (Q == I)
    contribution = 1.0*getRDM(J,P,Q,K,L,P, rdm );

  else if (P == J)
    contribution = 1.0*getRDM(I,P,Q,K,L,Q, rdm );

  else if (P == I)
    contribution = -1.0*getRDM(J,P,Q,K,L,Q, rdm );

  else if (P == Q)
    contribution = 1.0*getRDM(I,J,P,K,L,Q, rdm );

  else
    contribution = -1.0*getRDM(I,J,P,Q,K,L,P,Q, rdm );

  return contribution;  
}


complex<T> calcRDM::calcTerm2(int P, int Q, int I, int K, int R, int S, MatrixXcT& rdm) {
  complex<T> contribution = 0;
  
  if (P == Q && Q == I && K == R && R == S)
    contribution = 1.0*getRDM(P,S, rdm );

  else if (Q == I && K == R && R == S)
    contribution = -1.0*getRDM(P,Q,P,S, rdm );

  else if (P == S && Q == I && K == R)
    contribution = 1.0*getRDM(P,Q,R,S, rdm );

  else if (P == R && Q == I && R == S)
    contribution = 1.0*getRDM(P,Q,K,S, rdm );

  else if (P == R && Q == I && K == S)
    contribution = -1.0*getRDM(P,Q,R,S, rdm );

  else if (P == I && K == R && R == S)
    contribution = 1.0*getRDM(P,Q,Q,S, rdm );

  else if (P == I && Q == S && K == R)
    contribution = -1.0*getRDM(P,Q,R,S, rdm );

  else if (P == I && Q == R && R == S)
    contribution = -1.0*getRDM(P,Q,K,S, rdm );

  else if (P == I && Q == R && K == S)
    contribution = 1.0*getRDM(P,Q,R,S, rdm );

  else if (P == Q && K == R && R == S)
    contribution = 1.0*getRDM(I,P,Q,S, rdm );

  else if (P == Q && Q == S && K == R)
    contribution = -1.0*getRDM(I,P,R,S, rdm );

  else if (P == Q && Q == R && R == S)
    contribution = -1.0*getRDM(I,P,K,S, rdm );

  else if (P == Q && Q == R && K == S)
    contribution = 1.0*getRDM(I,P,R,S, rdm );

  else if (P == Q && Q == I && R == S)
    contribution = -1.0*getRDM(P,R,K,S, rdm );

  else if (P == Q && Q == I && K == S)
    contribution = 1.0*getRDM(P,R,R,S, rdm );

  else if (P == Q && Q == I && K == R)
    contribution = -1.0*getRDM(P,S,R,S, rdm );

  else if (K == R && R == S)
    contribution = -1.0*getRDM(I,P,Q,P,Q,S, rdm );

  else if (Q == S && K == R)
    contribution = 1.0*getRDM(I,P,Q,P,R,S, rdm );

  else if (Q == R && R == S)
    contribution = -1.0*getRDM(I,P,Q,K,P,S, rdm );

  else if (Q == R && K == S)
    contribution = -1.0*getRDM(I,P,Q,P,R,S, rdm );

  else if (Q == I && R == S)
    contribution = 1.0*getRDM(P,Q,R,K,P,S, rdm );

  else if (Q == I && K == S)
    contribution = 1.0*getRDM(P,Q,R,P,R,S, rdm );

  else if (Q == I && K == R)
    contribution = -1.0*getRDM(P,Q,S,P,R,S, rdm );

  else if (P == S && K == R)
    contribution = -1.0*getRDM(I,P,Q,Q,R,S, rdm );

  else if (P == S && Q == R)
    contribution = 1.0*getRDM(I,P,Q,K,R,S, rdm );

  else if (P == S && Q == I)
    contribution = -1.0*getRDM(P,Q,R,K,R,S, rdm );

  else if (P == R && R == S)
    contribution = 1.0*getRDM(I,P,Q,K,Q,S, rdm );

  else if (P == R && K == S)
    contribution = 1.0*getRDM(I,P,Q,Q,R,S, rdm );

  else if (P == R && Q == S)
    contribution = -1.0*getRDM(I,P,Q,K,R,S, rdm );

  else if (P == R && Q == I)
    contribution = 1.0*getRDM(P,Q,S,K,R,S, rdm );

  else if (P == I && R == S)
    contribution = -1.0*getRDM(P,Q,R,K,Q,S, rdm );

  else if (P == I && K == S)
    contribution = -1.0*getRDM(P,Q,R,Q,R,S, rdm );

  else if (P == I && K == R)
    contribution = 1.0*getRDM(P,Q,S,Q,R,S, rdm );

  else if (P == I && Q == S)
    contribution = 1.0*getRDM(P,Q,R,K,R,S, rdm );

  else if (P == I && Q == R)
    contribution = -1.0*getRDM(P,Q,S,K,R,S, rdm );

  else if (P == Q && R == S)
    contribution = -1.0*getRDM(I,P,R,K,Q,S, rdm );

  else if (P == Q && K == S)
    contribution = -1.0*getRDM(I,P,R,Q,R,S, rdm );

  else if (P == Q && K == R)
    contribution = 1.0*getRDM(I,P,S,Q,R,S, rdm );

  else if (P == Q && Q == S)
    contribution = 1.0*getRDM(I,P,R,K,R,S, rdm );

  else if (P == Q && Q == R)
    contribution = -1.0*getRDM(I,P,S,K,R,S, rdm );

  else if (P == Q && Q == I)
    contribution = -1.0*getRDM(P,R,S,K,R,S, rdm );

  else if (R == S)
    contribution = 1.0*getRDM(I,P,Q,R,K,P,Q,S, rdm );

  else if (K == S)
    contribution = -1.0*getRDM(I,P,Q,R,P,Q,R,S, rdm );

  else if (K == R)
    contribution = 1.0*getRDM(I,P,Q,S,P,Q,R,S, rdm );

  else if (Q == S)
    contribution = -1.0*getRDM(I,P,Q,R,K,P,R,S, rdm );

  else if (Q == R)
    contribution = 1.0*getRDM(I,P,Q,S,K,P,R,S, rdm );

  else if (Q == I)
    contribution = -1.0*getRDM(P,Q,R,S,K,P,R,S, rdm );

  else if (P == S)
    contribution = 1.0*getRDM(I,P,Q,R,K,Q,R,S, rdm );

  else if (P == R)
    contribution = -1.0*getRDM(I,P,Q,S,K,Q,R,S, rdm );

  else if (P == I)
    contribution = 1.0*getRDM(P,Q,R,S,K,Q,R,S, rdm );

  else if (P == Q)
    contribution = 1.0*getRDM(I,P,R,S,K,Q,R,S, rdm );

  else
    contribution = 1.0*getRDM(I,P,Q,R,S,K,P,Q,R,S, rdm );

  return contribution;
  
}


complex<T> calcRDM::calcTerm2(int P, int Q, int I, int J, int K, int L, int R, int S, MatrixXcT& rdm){
  if (K == -1 && L == -1)
    calcTerm2(P, Q, I, J, R, S, rdm);
  complex<T> contribution = 0.0;
  if (P == J && Q == I && L == R && R == S)
    contribution = -1.0*getRDM(P,Q,K,S, rdm );

  else if (P == J && Q == I && K == S && L == R)
    contribution = 1.0*getRDM(P,Q,R,S, rdm );

  else if (P == J && Q == I && K == R && R == S)
    contribution = 1.0*getRDM(P,Q,L,S, rdm );

  else if (P == J && Q == I && K == R && L == S)
    contribution = -1.0*getRDM(P,Q,R,S, rdm );

  else if (P == I && Q == J && L == R && R == S)
    contribution = 1.0*getRDM(P,Q,K,S, rdm );

  else if (P == I && Q == J && K == S && L == R)
    contribution = -1.0*getRDM(P,Q,R,S, rdm );

  else if (P == I && Q == J && K == R && R == S)
    contribution = -1.0*getRDM(P,Q,L,S, rdm );

  else if (P == I && Q == J && K == R && L == S)
    contribution = 1.0*getRDM(P,Q,R,S, rdm );

  else if (P == Q && Q == J && L == R && R == S)
    contribution = 1.0*getRDM(I,P,K,S, rdm );

  else if (P == Q && Q == J && K == S && L == R)
    contribution = -1.0*getRDM(I,P,R,S, rdm );

  else if (P == Q && Q == J && K == R && R == S)
    contribution = -1.0*getRDM(I,P,L,S, rdm );

  else if (P == Q && Q == J && K == R && L == S)
    contribution = 1.0*getRDM(I,P,R,S, rdm );

  else if (P == Q && Q == I && L == R && R == S)
    contribution = -1.0*getRDM(J,P,K,S, rdm );

  else if (P == Q && Q == I && K == S && L == R)
    contribution = 1.0*getRDM(J,P,R,S, rdm );

  else if (P == Q && Q == I && K == R && R == S)
    contribution = 1.0*getRDM(J,P,L,S, rdm );

  else if (P == Q && Q == I && K == R && L == S)
    contribution = -1.0*getRDM(J,P,R,S, rdm );

  else if (Q == J && L == R && R == S)
    contribution = 1.0*getRDM(I,P,Q,K,P,S, rdm );

  else if (Q == J && K == S && L == R)
    contribution = 1.0*getRDM(I,P,Q,P,R,S, rdm );

  else if (Q == J && K == R && R == S)
    contribution = -1.0*getRDM(I,P,Q,L,P,S, rdm );

  else if (Q == J && K == R && L == S)
    contribution = -1.0*getRDM(I,P,Q,P,R,S, rdm );

  else if (Q == I && L == R && R == S)
    contribution = -1.0*getRDM(J,P,Q,K,P,S, rdm );

  else if (Q == I && K == S && L == R)
    contribution = -1.0*getRDM(J,P,Q,P,R,S, rdm );

  else if (Q == I && K == R && R == S)
    contribution = 1.0*getRDM(J,P,Q,L,P,S, rdm );

  else if (Q == I && K == R && L == S)
    contribution = 1.0*getRDM(J,P,Q,P,R,S, rdm );

  else if (P == S && Q == J && L == R)
    contribution = -1.0*getRDM(I,P,Q,K,R,S, rdm );

  else if (P == S && Q == J && K == R)
    contribution = 1.0*getRDM(I,P,Q,L,R,S, rdm );

  else if (P == S && Q == I && L == R)
    contribution = 1.0*getRDM(J,P,Q,K,R,S, rdm );

  else if (P == S && Q == I && K == R)
    contribution = -1.0*getRDM(J,P,Q,L,R,S, rdm );

  else if (P == R && Q == J && R == S)
    contribution = -1.0*getRDM(I,P,Q,K,L,S, rdm );

  else if (P == R && Q == J && L == S)
    contribution = 1.0*getRDM(I,P,Q,K,R,S, rdm );

  else if (P == R && Q == J && K == S)
    contribution = -1.0*getRDM(I,P,Q,L,R,S, rdm );

  else if (P == R && Q == I && R == S)
    contribution = 1.0*getRDM(J,P,Q,K,L,S, rdm );

  else if (P == R && Q == I && L == S)
    contribution = -1.0*getRDM(J,P,Q,K,R,S, rdm );

  else if (P == R && Q == I && K == S)
    contribution = 1.0*getRDM(J,P,Q,L,R,S, rdm );

  else if (P == J && L == R && R == S)
    contribution = -1.0*getRDM(I,P,Q,K,Q,S, rdm );

  else if (P == J && K == S && L == R)
    contribution = -1.0*getRDM(I,P,Q,Q,R,S, rdm );

  else if (P == J && K == R && R == S)
    contribution = 1.0*getRDM(I,P,Q,L,Q,S, rdm );

  else if (P == J && K == R && L == S)
    contribution = 1.0*getRDM(I,P,Q,Q,R,S, rdm );

  else if (P == J && Q == S && L == R)
    contribution = 1.0*getRDM(I,P,Q,K,R,S, rdm );

  else if (P == J && Q == S && K == R)
    contribution = -1.0*getRDM(I,P,Q,L,R,S, rdm );

  else if (P == J && Q == R && R == S)
    contribution = 1.0*getRDM(I,P,Q,K,L,S, rdm );

  else if (P == J && Q == R && L == S)
    contribution = -1.0*getRDM(I,P,Q,K,R,S, rdm );

  else if (P == J && Q == R && K == S)
    contribution = 1.0*getRDM(I,P,Q,L,R,S, rdm );

  else if (P == J && Q == I && R == S)
    contribution = -1.0*getRDM(P,Q,R,K,L,S, rdm );

  else if (P == J && Q == I && L == S)
    contribution = 1.0*getRDM(P,Q,R,K,R,S, rdm );

  else if (P == J && Q == I && L == R)
    contribution = -1.0*getRDM(P,Q,S,K,R,S, rdm );

  else if (P == J && Q == I && K == S)
    contribution = -1.0*getRDM(P,Q,R,L,R,S, rdm );

  else if (P == J && Q == I && K == R)
    contribution = 1.0*getRDM(P,Q,S,L,R,S, rdm );

  else if (P == I && L == R && R == S)
    contribution = 1.0*getRDM(J,P,Q,K,Q,S, rdm );

  else if (P == I && K == S && L == R)
    contribution = 1.0*getRDM(J,P,Q,Q,R,S, rdm );

  else if (P == I && K == R && R == S)
    contribution = -1.0*getRDM(J,P,Q,L,Q,S, rdm );

  else if (P == I && K == R && L == S)
    contribution = -1.0*getRDM(J,P,Q,Q,R,S, rdm );

  else if (P == I && Q == S && L == R)
    contribution = -1.0*getRDM(J,P,Q,K,R,S, rdm );

  else if (P == I && Q == S && K == R)
    contribution = 1.0*getRDM(J,P,Q,L,R,S, rdm );

  else if (P == I && Q == R && R == S)
    contribution = -1.0*getRDM(J,P,Q,K,L,S, rdm );

  else if (P == I && Q == R && L == S)
    contribution = 1.0*getRDM(J,P,Q,K,R,S, rdm );

  else if (P == I && Q == R && K == S)
    contribution = -1.0*getRDM(J,P,Q,L,R,S, rdm );

  else if (P == I && Q == J && R == S)
    contribution = 1.0*getRDM(P,Q,R,K,L,S, rdm );

  else if (P == I && Q == J && L == S)
    contribution = -1.0*getRDM(P,Q,R,K,R,S, rdm );

  else if (P == I && Q == J && L == R)
    contribution = 1.0*getRDM(P,Q,S,K,R,S, rdm );

  else if (P == I && Q == J && K == S)
    contribution = 1.0*getRDM(P,Q,R,L,R,S, rdm );

  else if (P == I && Q == J && K == R)
    contribution = -1.0*getRDM(P,Q,S,L,R,S, rdm );

  else if (P == Q && L == R && R == S)
    contribution = -1.0*getRDM(I,J,P,K,Q,S, rdm );

  else if (P == Q && K == S && L == R)
    contribution = -1.0*getRDM(I,J,P,Q,R,S, rdm );

  else if (P == Q && K == R && R == S)
    contribution = 1.0*getRDM(I,J,P,L,Q,S, rdm );

  else if (P == Q && K == R && L == S)
    contribution = 1.0*getRDM(I,J,P,Q,R,S, rdm );

  else if (P == Q && Q == S && L == R)
    contribution = 1.0*getRDM(I,J,P,K,R,S, rdm );

  else if (P == Q && Q == S && K == R)
    contribution = -1.0*getRDM(I,J,P,L,R,S, rdm );

  else if (P == Q && Q == R && R == S)
    contribution = 1.0*getRDM(I,J,P,K,L,S, rdm );

  else if (P == Q && Q == R && L == S)
    contribution = -1.0*getRDM(I,J,P,K,R,S, rdm );

  else if (P == Q && Q == R && K == S)
    contribution = 1.0*getRDM(I,J,P,L,R,S, rdm );

  else if (P == Q && Q == J && R == S)
    contribution = 1.0*getRDM(I,P,R,K,L,S, rdm );

  else if (P == Q && Q == J && L == S)
    contribution = -1.0*getRDM(I,P,R,K,R,S, rdm );

  else if (P == Q && Q == J && L == R)
    contribution = 1.0*getRDM(I,P,S,K,R,S, rdm );

  else if (P == Q && Q == J && K == S)
    contribution = 1.0*getRDM(I,P,R,L,R,S, rdm );

  else if (P == Q && Q == J && K == R)
    contribution = -1.0*getRDM(I,P,S,L,R,S, rdm );

  else if (P == Q && Q == I && R == S)
    contribution = -1.0*getRDM(J,P,R,K,L,S, rdm );

  else if (P == Q && Q == I && L == S)
    contribution = 1.0*getRDM(J,P,R,K,R,S, rdm );

  else if (P == Q && Q == I && L == R)
    contribution = -1.0*getRDM(J,P,S,K,R,S, rdm );

  else if (P == Q && Q == I && K == S)
    contribution = -1.0*getRDM(J,P,R,L,R,S, rdm );

  else if (P == Q && Q == I && K == R)
    contribution = 1.0*getRDM(J,P,S,L,R,S, rdm );

  else if (L == R && R == S)
    contribution = -1.0*getRDM(I,J,P,Q,K,P,Q,S, rdm );

  else if (K == S && L == R)
    contribution = 1.0*getRDM(I,J,P,Q,P,Q,R,S, rdm );

  else if (K == R && R == S)
    contribution = 1.0*getRDM(I,J,P,Q,L,P,Q,S, rdm );

  else if (K == R && L == S)
    contribution = -1.0*getRDM(I,J,P,Q,P,Q,R,S, rdm );

  else if (Q == S && L == R)
    contribution = 1.0*getRDM(I,J,P,Q,K,P,R,S, rdm );

  else if (Q == S && K == R)
    contribution = -1.0*getRDM(I,J,P,Q,L,P,R,S, rdm );

  else if (Q == R && R == S)
    contribution = -1.0*getRDM(I,J,P,Q,K,L,P,S, rdm );

  else if (Q == R && L == S)
    contribution = -1.0*getRDM(I,J,P,Q,K,P,R,S, rdm );

  else if (Q == R && K == S)
    contribution = 1.0*getRDM(I,J,P,Q,L,P,R,S, rdm );

  else if (Q == J && R == S)
    contribution = 1.0*getRDM(I,P,Q,R,K,L,P,S, rdm );

  else if (Q == J && L == S)
    contribution = 1.0*getRDM(I,P,Q,R,K,P,R,S, rdm );

  else if (Q == J && L == R)
    contribution = -1.0*getRDM(I,P,Q,S,K,P,R,S, rdm );

  else if (Q == J && K == S)
    contribution = -1.0*getRDM(I,P,Q,R,L,P,R,S, rdm );

  else if (Q == J && K == R)
    contribution = 1.0*getRDM(I,P,Q,S,L,P,R,S, rdm );

  else if (Q == I && R == S)
    contribution = -1.0*getRDM(J,P,Q,R,K,L,P,S, rdm );

  else if (Q == I && L == S)
    contribution = -1.0*getRDM(J,P,Q,R,K,P,R,S, rdm );

  else if (Q == I && L == R)
    contribution = 1.0*getRDM(J,P,Q,S,K,P,R,S, rdm );

  else if (Q == I && K == S)
    contribution = 1.0*getRDM(J,P,Q,R,L,P,R,S, rdm );

  else if (Q == I && K == R)
    contribution = -1.0*getRDM(J,P,Q,S,L,P,R,S, rdm );

  else if (P == S && L == R)
    contribution = -1.0*getRDM(I,J,P,Q,K,Q,R,S, rdm );

  else if (P == S && K == R)
    contribution = 1.0*getRDM(I,J,P,Q,L,Q,R,S, rdm );

  else if (P == S && Q == R)
    contribution = 1.0*getRDM(I,J,P,Q,K,L,R,S, rdm );

  else if (P == S && Q == J)
    contribution = -1.0*getRDM(I,P,Q,R,K,L,R,S, rdm );

  else if (P == S && Q == I)
    contribution = 1.0*getRDM(J,P,Q,R,K,L,R,S, rdm );

  else if (P == R && R == S)
    contribution = 1.0*getRDM(I,J,P,Q,K,L,Q,S, rdm );

  else if (P == R && L == S)
    contribution = 1.0*getRDM(I,J,P,Q,K,Q,R,S, rdm );

  else if (P == R && K == S)
    contribution = -1.0*getRDM(I,J,P,Q,L,Q,R,S, rdm );

  else if (P == R && Q == S)
    contribution = -1.0*getRDM(I,J,P,Q,K,L,R,S, rdm );

  else if (P == R && Q == J)
    contribution = 1.0*getRDM(I,P,Q,S,K,L,R,S, rdm );

  else if (P == R && Q == I)
    contribution = -1.0*getRDM(J,P,Q,S,K,L,R,S, rdm );

  else if (P == J && R == S)
    contribution = -1.0*getRDM(I,P,Q,R,K,L,Q,S, rdm );

  else if (P == J && L == S)
    contribution = -1.0*getRDM(I,P,Q,R,K,Q,R,S, rdm );

  else if (P == J && L == R)
    contribution = 1.0*getRDM(I,P,Q,S,K,Q,R,S, rdm );

  else if (P == J && K == S)
    contribution = 1.0*getRDM(I,P,Q,R,L,Q,R,S, rdm );

  else if (P == J && K == R)
    contribution = -1.0*getRDM(I,P,Q,S,L,Q,R,S, rdm );

  else if (P == J && Q == S)
    contribution = 1.0*getRDM(I,P,Q,R,K,L,R,S, rdm );

  else if (P == J && Q == R)
    contribution = -1.0*getRDM(I,P,Q,S,K,L,R,S, rdm );

  else if (P == J && Q == I)
    contribution = 1.0*getRDM(P,Q,R,S,K,L,R,S, rdm );

  else if (P == I && R == S)
    contribution = 1.0*getRDM(J,P,Q,R,K,L,Q,S, rdm );

  else if (P == I && L == S)
    contribution = 1.0*getRDM(J,P,Q,R,K,Q,R,S, rdm );

  else if (P == I && L == R)
    contribution = -1.0*getRDM(J,P,Q,S,K,Q,R,S, rdm );

  else if (P == I && K == S)
    contribution = -1.0*getRDM(J,P,Q,R,L,Q,R,S, rdm );

  else if (P == I && K == R)
    contribution = 1.0*getRDM(J,P,Q,S,L,Q,R,S, rdm );

  else if (P == I && Q == S)
    contribution = -1.0*getRDM(J,P,Q,R,K,L,R,S, rdm );

  else if (P == I && Q == R)
    contribution = 1.0*getRDM(J,P,Q,S,K,L,R,S, rdm );

  else if (P == I && Q == J)
    contribution = -1.0*getRDM(P,Q,R,S,K,L,R,S, rdm );

  else if (P == Q && R == S)
    contribution = -1.0*getRDM(I,J,P,R,K,L,Q,S, rdm );

  else if (P == Q && L == S)
    contribution = -1.0*getRDM(I,J,P,R,K,Q,R,S, rdm );

  else if (P == Q && L == R)
    contribution = 1.0*getRDM(I,J,P,S,K,Q,R,S, rdm );

  else if (P == Q && K == S)
    contribution = 1.0*getRDM(I,J,P,R,L,Q,R,S, rdm );

  else if (P == Q && K == R)
    contribution = -1.0*getRDM(I,J,P,S,L,Q,R,S, rdm );

  else if (P == Q && Q == S)
    contribution = 1.0*getRDM(I,J,P,R,K,L,R,S, rdm );

  else if (P == Q && Q == R)
    contribution = -1.0*getRDM(I,J,P,S,K,L,R,S, rdm );

  else if (P == Q && Q == J)
    contribution = -1.0*getRDM(I,P,R,S,K,L,R,S, rdm );

  else if (P == Q && Q == I)
    contribution = 1.0*getRDM(J,P,R,S,K,L,R,S, rdm );

  else if (R == S)
    contribution = -1.0*getRDM(I,J,P,Q,R,K,L,P,Q,S, rdm );

  else if (L == S)
    contribution = 1.0*getRDM(I,J,P,Q,R,K,P,Q,R,S, rdm );

  else if (L == R)
    contribution = -1.0*getRDM(I,J,P,Q,S,K,P,Q,R,S, rdm );

  else if (K == S)
    contribution = -1.0*getRDM(I,J,P,Q,R,L,P,Q,R,S, rdm );

  else if (K == R)
    contribution = 1.0*getRDM(I,J,P,Q,S,L,P,Q,R,S, rdm );

  else if (Q == S)
    contribution = 1.0*getRDM(I,J,P,Q,R,K,L,P,R,S, rdm );

  else if (Q == R)
    contribution = -1.0*getRDM(I,J,P,Q,S,K,L,P,R,S, rdm );

  else if (Q == J)
    contribution = 1.0*getRDM(I,P,Q,R,S,K,L,P,R,S, rdm );

  else if (Q == I)
    contribution = -1.0*getRDM(J,P,Q,R,S,K,L,P,R,S, rdm );

  else if (P == S)
    contribution = -1.0*getRDM(I,J,P,Q,R,K,L,Q,R,S, rdm );

  else if (P == R)
    contribution = 1.0*getRDM(I,J,P,Q,S,K,L,Q,R,S, rdm );

  else if (P == J)
    contribution = -1.0*getRDM(I,P,Q,R,S,K,L,Q,R,S, rdm );

  else if (P == I)
    contribution = 1.0*getRDM(J,P,Q,R,S,K,L,Q,R,S, rdm );

  else if (P == Q)
    contribution = -1.0*getRDM(I,J,P,R,S,K,L,Q,R,S, rdm );

  else
    contribution = 1.0*getRDM(I,J,P,Q,R,S,K,L,P,Q,R,S, rdm );

  return contribution;
  
}
