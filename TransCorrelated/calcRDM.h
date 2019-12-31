#pragma once
#include <Eigen/Dense>
#include <vector>
using namespace std;
using namespace Eigen;

template <
  typename DerivedX,
  typename DerivedR,
  typename DerivedC,
  typename DerivedY>
void Slice(
    const Eigen::DenseBase<DerivedX> & X,
    const vector<DerivedR> & R,
    const vector<DerivedC> & C,
    DerivedY & Y)
{
  int ym = Y.rows();
  int yn = Y.cols();

  // loop over output rows, then columns
  for(int i = 0;i<ym;i++)
  {
    for(int j = 0;j<yn;j++)
    {
      Y(i,j) = X(R[i],C[j]);
    }
  }
};

template<typename T, typename complexT>
class calcRDM{
  using Matrix2cT = Eigen::Matrix<complexT, 2, 2>;
  using Matrix3cT = Eigen::Matrix<complexT, 3, 3>;
  using Matrix4cT = Eigen::Matrix<complexT, 4, 4>;
  using MatrixXcT = Eigen::Matrix<complexT, Eigen::Dynamic, Eigen::Dynamic>;

  Matrix2cT rdmval2;
  Matrix3cT rdmval3;
  Matrix4cT rdmval4;
  MatrixXcT rdmval5;
  MatrixXcT rdmval6;
  MatrixXcT rdmval7;
  MatrixXcT rdmval8;

  vector<int> rows, cols;
 public:
  calcRDM() {
    rows.resize(8,0); cols.resize(8,0);
    rdmval5.resize(5,5);
    rdmval6.resize(6,6);
    rdmval7.resize(7,7);
    rdmval8.resize(8,8);
  }
  
  complexT getRDM(int a, int b, MatrixXcT& rdm) {
    return rdm(b, a);
  }

  complexT getRDM(int a, int b, int c, int d, MatrixXcT& rdm){
    rows[0] = d; rows[1] = c;
    cols[0] = a; cols[1] = b;
    Slice(rdm, rows, cols, rdmval2);
    return rdmval2.determinant();
  }

  complexT getRDM(int a, int b, int c, int d, int e, int f, MatrixXcT& rdm){
    rows[0] = f; rows[1] = e; rows[2] = d;
    cols[0] = a; cols[1] = b; cols[2] = c;
    Slice(rdm, rows, cols, rdmval3);
    return rdmval3.determinant();
  }

  complexT getRDM(int a, int b, int c, int d, int e, int f, int g, int h, MatrixXcT& rdm){
    rows[0] = h; rows[1] = g; rows[2] = f; rows[3] = e;
    cols[0] = a; cols[1] = b; cols[2] = c; cols[3] = d; 
    Slice(rdm, rows, cols, rdmval4);
    return rdmval4.determinant();
  }

  complexT getRDM(int a, int b, int c, int d, int e, int f, int g, int h, int i, int j, MatrixXcT& rdm){
    rows[0] = j; rows[1] = i; rows[2] = h; rows[3] = g; rows[4] = f;
    cols[0] = a; cols[1] = b; cols[2] = c; cols[3] = d; cols[4] = e; 
    Slice(rdm, rows, cols, rdmval5);
    return rdmval5.determinant();
  }

  complexT getRDM(int a, int b, int c, int d, int e, int f, int g, int h, int i, int j, int k, int l, MatrixXcT& rdm){
    rows[0] = l; rows[1] = k; rows[2] = j; rows[3] = i; rows[4] = h; rows[5] = g;
    cols[0] = a; cols[1] = b; cols[2] = c; cols[3] = d; cols[4] = e; cols[5] = f;
    Slice(rdm, rows, cols, rdmval6);
    return rdmval6.determinant();
  }

  complexT getRDM(int a, int b, int c, int d, int e, int f, int g, int h, int i, int j, int k, int l, int m, int n, MatrixXcT& rdm){
    rows[0] = n; rows[1] = m; rows[2] = l; rows[3] = k; rows[4] = j; rows[5] = i; rows[6] = h;
    cols[0] = a; cols[1] = b; cols[2] = c; cols[3] = d; cols[4] = e; cols[5] = f; cols[6] = g;
    Slice(rdm, rows, cols, rdmval7);
    return rdmval7.determinant();
  }

  complexT getRDM(int a, int b, int c, int d, int e, int f, int g, int h, int i, int j, int k, int l, int m, int n, int o, int p, MatrixXcT& rdm){
    rows[0] = p; rows[1] = o; rows[2] = n; rows[3] = m; rows[4] = l; rows[5] = k; rows[6] = j; rows[7] = i;
    cols[0] = a; cols[1] = b; cols[2] = c; cols[3] = d; cols[4] = e; cols[5] = f; cols[6] = g; cols[7] = h;
    Slice(rdm, rows, cols, rdmval8);
    return rdmval8.determinant();
  }

  //computer generated
  complexT calcTermGJ1(int I, int K, MatrixXcT& rdm,
                       int P, int Q, T xiP, T xiQ){
    //P = Ispatial + opposite spin
    //Q = Jspatial + opposite spin

    complexT contribution = complexT(0., 0.);
    contribution += xiP * getRDM(P,I,K,P, rdm );
    contribution += xiQ * getRDM(Q,I,K,Q, rdm );

    complexT contributionik = complexT(0, 0);
    if (P == I && K == Q)
      contributionik += 1.0*getRDM(P,Q, rdm );

    else if (K == Q)
      contributionik += 1.0*getRDM(I,P,P,Q, rdm );

    else if (P == Q)
      contributionik += -1.0*getRDM(I,P,K,Q, rdm );

    else if (P == I)
      contributionik += -1.0*getRDM(P,Q,K,Q, rdm );

    else
      contributionik += -1.0*getRDM(I,P,Q,K,P,Q, rdm );

    return contribution + xiP * xiQ * contributionik;
  }

  complexT calcTermGJ1jas(int N, int M, int I, int K, MatrixXcT& rdm,
                          int P, int Q, T xiP, T xiQ) {
    complexT contribution = complexT(0., 0.);

    contribution  = calcTermGJ1jasA(N, M, I, K, rdm);                  
    contribution += xiP * calcTermGJ1jasB(N, M, P, I, K, rdm);         
    contribution += xiQ * calcTermGJ1jasC(N, M, I, K, Q, rdm);         
    contribution += xiP * xiQ * calcTermGJ1jasD(N, M, P, I, K, Q, rdm);
    return contribution;
  }
  
  //computer generated
  complexT calcTermGJ2(int I, int J, int K, int L, MatrixXcT& rdm,
                       int P, int Q, int R, int S,
                       double xiP, double xiQ, double xiR, double xiS){

    complexT contribution = complexT(0., 0.);
    //there are 16 terms
    contribution  = getRDM(I,J,K,L, rdm);
    contribution += xiP * calcTermGJ2a(P, I,J,K,L, rdm);
    contribution += xiQ * calcTermGJ2a(Q, I,J,K,L, rdm);
    contribution += xiR * calcTermGJ2b(I,J,K,L, R, rdm);
    contribution += xiS * calcTermGJ2b(I,J,K,L, S, rdm);

    contribution += xiP * xiQ * calcTermGJ2c(P, Q, I, J, K, L, rdm);
    contribution += xiP * xiR * calcTermGJ2d(P, I, J, K, L, R, rdm);
    contribution += xiP * xiS * calcTermGJ2d(P, I, J, K, L, S, rdm);
    contribution += xiQ * xiR * calcTermGJ2d(Q, I, J, K, L, R, rdm);
    contribution += xiQ * xiS * calcTermGJ2d(Q, I, J, K, L, S, rdm);
    contribution += xiR * xiS * calcTermGJ2e(I, J, K, L, R, S, rdm);
    
    contribution += xiP * xiQ * xiR * calcTermGJ2f(P, Q, I, J, K, L, R, rdm);
    contribution += xiP * xiQ * xiS * calcTermGJ2f(P, Q, I, J, K, L, S, rdm);
    contribution += xiP * xiR * xiS * calcTermGJ2g(P, I, J, K, L, R, S, rdm);
    contribution += xiQ * xiR * xiS * calcTermGJ2g(Q, I, J, K, L, R, S, rdm);

    contribution += xiP * xiQ * xiR * xiS * calcTermGJ2h(P, Q, I, J, K, L, R, S, rdm);

    return contribution;
  }

    //computer generated
  complexT calcTermGJ2jas(int M, int N, int I, int J, int K, int L, MatrixXcT& rdm,
                          int P, int Q, int R, int S,
                          double xiP, double xiQ, double xiR, double xiS){

    complexT contribution = complexT(0., 0.);
    //there are 16 terms
    contribution  = calcTermGJ2c(M, N, I,J,K,L, rdm);   //already implemented
    contribution += xiP * calcTermGJ2i(M, N, P, I,J,K, L, rdm);
    contribution += xiQ * calcTermGJ2i(M, N, Q, I,J,K, L, rdm);
    contribution += xiR * calcTermGJ2f(M, N, I, J,K,L, R, rdm);
    contribution += xiS * calcTermGJ2f(M, N, I, J,K,L, S, rdm);

    contribution += xiP * xiQ * calcTermGJ2j(M, N, P, Q, I, J, K, L, rdm);
    contribution += xiP * xiR * calcTermGJ2k(M, N, P, I, J, K, L, R, rdm);
    contribution += xiP * xiS * calcTermGJ2k(M, N, P, I, J, K, L, S, rdm);
    contribution += xiQ * xiR * calcTermGJ2k(M, N, Q, I, J, K, L, R, rdm);
    contribution += xiQ * xiS * calcTermGJ2k(M, N, Q, I, J, K, L, S, rdm);
    contribution += xiR * xiS * calcTermGJ2h(M, N, I, J, K, L, R, S, rdm);
    
    contribution += xiP * xiQ * xiR * calcTermGJ2l(M, N, P, Q, I, J, K, L, R, rdm);
    contribution += xiP * xiQ * xiS * calcTermGJ2l(M, N, P, Q, I, J, K, L, S, rdm);
    contribution += xiP * xiR * xiS * calcTermGJ2m(M, N, P, I, J, K, L, R, S, rdm);
    contribution += xiQ * xiR * xiS * calcTermGJ2m(M, N, Q, I, J, K, L, R, S, rdm);

    contribution += xiP * xiQ * xiR * xiS * calcTermGJ2n(M, N, P, Q, I, J, K, L, R, S, rdm);

    return contribution;
  }

  complexT calcTermGJ1jasA(int N, int M, int I, int K, MatrixXcT& rdm) {
    complexT contribution = complexT(0.,0.);
    if (M == N && N == I)
      contribution += 1.0*getRDM(M,K, rdm );

    else if (N == I)
      contribution += 1.0*getRDM(M,N,K,M, rdm );

    else if (M == I)
      contribution += -1.0*getRDM(M,N,K,N, rdm );

    else if (M == N)
      contribution += -1.0*getRDM(I,M,K,N, rdm );

    else
      contribution += -1.0*getRDM(I,M,N,K,M,N, rdm );

    return contribution;
  }
  
  complexT calcTermGJ1jasB(int N, int M, int P, int I, int K, MatrixXcT& rdm) {
    complexT contribution = complexT(0.,0.);
    if (M == N && N == P && P == I)
      contribution += 1.0*getRDM(M,K, rdm );

    else if (N == P && P == I)
      contribution += 1.0*getRDM(M,N,K,M, rdm );

    else if (M == I && N == P)
      contribution += -1.0*getRDM(M,N,K,P, rdm );

    else if (M == P && P == I)
      contribution += -1.0*getRDM(M,N,K,N, rdm );

    else if (M == P && N == I)
      contribution += 1.0*getRDM(M,N,K,P, rdm );

    else if (M == N && P == I)
      contribution += 1.0*getRDM(M,P,K,N, rdm );

    else if (M == N && N == I)
      contribution += -1.0*getRDM(M,P,K,P, rdm );

    else if (M == N && N == P)
      contribution += -1.0*getRDM(I,M,K,P, rdm );

    else if (P == I)
      contribution += -1.0*getRDM(M,N,P,K,M,N, rdm );

    else if (N == I)
      contribution += 1.0*getRDM(M,N,P,K,M,P, rdm );

    else if (N == P)
      contribution += -1.0*getRDM(I,M,N,K,M,P, rdm );

    else if (M == I)
      contribution += -1.0*getRDM(M,N,P,K,N,P, rdm );

    else if (M == P)
      contribution += 1.0*getRDM(I,M,N,K,N,P, rdm );

    else if (M == N)
      contribution += -1.0*getRDM(I,M,P,K,N,P, rdm );

    else
      contribution += 1.0*getRDM(I,M,N,P,K,M,N,P, rdm );

    return contribution;
    
  }
  
  complexT calcTermGJ1jasC(int N, int M, int I, int K, int Q, MatrixXcT& rdm) {
    complexT contribution = complexT(0.,0.);
    if (M == N && N == I && K == Q)
      contribution += 1.0*getRDM(M,Q, rdm );

    else if (N == I && K == Q)
      contribution += -1.0*getRDM(M,N,M,Q, rdm );

    else if (M == Q && N == I)
      contribution += 1.0*getRDM(M,N,K,Q, rdm );

    else if (M == I && K == Q)
      contribution += 1.0*getRDM(M,N,N,Q, rdm );

    else if (M == I && N == Q)
      contribution += -1.0*getRDM(M,N,K,Q, rdm );

    else if (M == N && K == Q)
      contribution += 1.0*getRDM(I,M,N,Q, rdm );

    else if (M == N && N == Q)
      contribution += -1.0*getRDM(I,M,K,Q, rdm );

    else if (M == N && N == I)
      contribution += -1.0*getRDM(M,Q,K,Q, rdm );

    else if (K == Q)
      contribution += -1.0*getRDM(I,M,N,M,N,Q, rdm );

    else if (N == Q)
      contribution += -1.0*getRDM(I,M,N,K,M,Q, rdm );

    else if (N == I)
      contribution += 1.0*getRDM(M,N,Q,K,M,Q, rdm );

    else if (M == Q)
      contribution += 1.0*getRDM(I,M,N,K,N,Q, rdm );

    else if (M == I)
      contribution += -1.0*getRDM(M,N,Q,K,N,Q, rdm );

    else if (M == N)
      contribution += -1.0*getRDM(I,M,Q,K,N,Q, rdm );

    else
      contribution += 1.0*getRDM(I,M,N,Q,K,M,N,Q, rdm );

    return contribution;
    
  }
  
  complexT calcTermGJ1jasD(int N, int M, int P, int I, int K, int Q, MatrixXcT& rdm) {
    complexT contribution = complexT(0.,0.);
    if (N == P && P == I && K == Q)
      contribution += -1.0*getRDM(M,N,M,Q, rdm );

    else if (M == Q && N == P && P == I)
      contribution += 1.0*getRDM(M,N,K,Q, rdm );

    else if (M == I && N == P && K == Q)
      contribution += 1.0*getRDM(M,N,P,Q, rdm );

    else if (M == I && N == P && P == Q)
      contribution += -1.0*getRDM(M,N,K,Q, rdm );

    else if (M == P && P == I && K == Q)
      contribution += 1.0*getRDM(M,N,N,Q, rdm );

    else if (M == P && N == Q && P == I)
      contribution += -1.0*getRDM(M,N,K,Q, rdm );

    else if (M == P && N == I && K == Q)
      contribution += -1.0*getRDM(M,N,P,Q, rdm );

    else if (M == P && N == I && P == Q)
      contribution += 1.0*getRDM(M,N,K,Q, rdm );

    else if (P == I && K == Q)
      contribution += -1.0*getRDM(M,N,P,M,N,Q, rdm );

    else if (N == Q && P == I)
      contribution += -1.0*getRDM(M,N,P,K,M,Q, rdm );

    else if (N == I && K == Q)
      contribution += 1.0*getRDM(M,N,P,M,P,Q, rdm );

    else if (N == I && P == Q)
      contribution += 1.0*getRDM(M,N,P,K,M,Q, rdm );

    else if (N == P && K == Q)
      contribution += -1.0*getRDM(I,M,N,M,P,Q, rdm );

    else if (N == P && P == Q)
      contribution += -1.0*getRDM(I,M,N,K,M,Q, rdm );

    else if (N == P && P == I)
      contribution += 1.0*getRDM(M,N,Q,K,M,Q, rdm );

    else if (M == Q && P == I)
      contribution += 1.0*getRDM(M,N,P,K,N,Q, rdm );

    else if (M == Q && N == I)
      contribution += -1.0*getRDM(M,N,P,K,P,Q, rdm );

    else if (M == Q && N == P)
      contribution += 1.0*getRDM(I,M,N,K,P,Q, rdm );

    else if (M == I && K == Q)
      contribution += -1.0*getRDM(M,N,P,N,P,Q, rdm );

    else if (M == I && P == Q)
      contribution += -1.0*getRDM(M,N,P,K,N,Q, rdm );

    else if (M == I && N == Q)
      contribution += 1.0*getRDM(M,N,P,K,P,Q, rdm );

    else if (M == I && N == P)
      contribution += -1.0*getRDM(M,N,Q,K,P,Q, rdm );

    else if (M == P && K == Q)
      contribution += 1.0*getRDM(I,M,N,N,P,Q, rdm );

    else if (M == P && P == Q)
      contribution += 1.0*getRDM(I,M,N,K,N,Q, rdm );

    else if (M == P && P == I)
      contribution += -1.0*getRDM(M,N,Q,K,N,Q, rdm );

    else if (M == P && N == Q)
      contribution += -1.0*getRDM(I,M,N,K,P,Q, rdm );

    else if (M == P && N == I)
      contribution += 1.0*getRDM(M,N,Q,K,P,Q, rdm );

    else if (K == Q)
      contribution += -1.0*getRDM(I,M,N,P,M,N,P,Q, rdm );

    else if (P == Q)
      contribution += 1.0*getRDM(I,M,N,P,K,M,N,Q, rdm );

    else if (P == I)
      contribution += 1.0*getRDM(M,N,P,Q,K,M,N,Q, rdm );

    else if (N == Q)
      contribution += -1.0*getRDM(I,M,N,P,K,M,P,Q, rdm );

    else if (N == I)
      contribution += -1.0*getRDM(M,N,P,Q,K,M,P,Q, rdm );

    else if (N == P)
      contribution += 1.0*getRDM(I,M,N,Q,K,M,P,Q, rdm );

    else if (M == Q)
      contribution += 1.0*getRDM(I,M,N,P,K,N,P,Q, rdm );

    else if (M == I)
      contribution += 1.0*getRDM(M,N,P,Q,K,N,P,Q, rdm );

    else if (M == P)
      contribution += -1.0*getRDM(I,M,N,Q,K,N,P,Q, rdm );

    else
      contribution += 1.0*getRDM(I,M,N,P,Q,K,M,N,P,Q, rdm );

    return contribution;
    
  }
  
  complexT calcTermGJ2a(int P, int I, int J, int K, int L, MatrixXcT& rdm) {
    complexT contribution = complexT(0.,0.);
    if (P == J)
      contribution += 1.0*getRDM(I,P,K,L, rdm );

    else if (P == I)
      contribution += -1.0*getRDM(J,P,K,L, rdm );

    else
      contribution += 1.0*getRDM(I,J,P,K,L,P, rdm );

    return contribution;
    
  }
  
  complexT calcTermGJ2b(int I, int J, int K, int L, int S, MatrixXcT& rdm) {
    complexT contribution = complexT(0.,0.);
    if (L == S)
      contribution += 1.0*getRDM(I,J,K,S, rdm );

    else if (K == S)
      contribution += -1.0*getRDM(I,J,L,S, rdm );

    else
      contribution += 1.0*getRDM(I,J,S,K,L,S, rdm );

    return contribution;
  }
  
  complexT calcTermGJ2c(int P, int Q, int I, int J, int K, int L, MatrixXcT& rdm) {
    complexT contribution = complexT(0.,0.);
    if (P == J && Q == I)
      contribution += -1.0*getRDM(P,Q,K,L, rdm );

    else if (P == I && Q == J)
      contribution += 1.0*getRDM(P,Q,K,L, rdm );

    else if (P == Q && Q == J)
      contribution += 1.0*getRDM(I,P,K,L, rdm );

    else if (P == Q && Q == I)
      contribution += -1.0*getRDM(J,P,K,L, rdm );

    else if (Q == J)
      contribution += -1.0*getRDM(I,P,Q,K,L,P, rdm );

    else if (Q == I)
      contribution += 1.0*getRDM(J,P,Q,K,L,P, rdm );

    else if (P == J)
      contribution += 1.0*getRDM(I,P,Q,K,L,Q, rdm );

    else if (P == I)
      contribution += -1.0*getRDM(J,P,Q,K,L,Q, rdm );

    else if (P == Q)
      contribution += 1.0*getRDM(I,J,P,K,L,Q, rdm );

    else
      contribution += -1.0*getRDM(I,J,P,Q,K,L,P,Q, rdm );

    return contribution;
  }
  
  complexT calcTermGJ2d(int P, int I, int J, int K, int L, int R, MatrixXcT& rdm) {
    complexT contribution = complexT(0.,0.);
    if (P == J && L == R)
      contribution += 1.0*getRDM(I,P,K,R, rdm );

    else if (P == J && K == R)
      contribution += -1.0*getRDM(I,P,L,R, rdm );

    else if (P == I && L == R)
      contribution += -1.0*getRDM(J,P,K,R, rdm );

    else if (P == I && K == R)
      contribution += 1.0*getRDM(J,P,L,R, rdm );

    else if (L == R)
      contribution += -1.0*getRDM(I,J,P,K,P,R, rdm );

    else if (K == R)
      contribution += 1.0*getRDM(I,J,P,L,P,R, rdm );

    else if (P == R)
      contribution += 1.0*getRDM(I,J,P,K,L,R, rdm );

    else if (P == J)
      contribution += 1.0*getRDM(I,P,R,K,L,R, rdm );

    else if (P == I)
      contribution += -1.0*getRDM(J,P,R,K,L,R, rdm );

    else
      contribution += -1.0*getRDM(I,J,P,R,K,L,P,R, rdm );

    return contribution;
  }
  
  complexT calcTermGJ2e(int I, int J, int K, int L, int R, int S, MatrixXcT& rdm) {
    complexT contribution = complexT(0.,0.);
    if (L == R && R == S)
      contribution += 1.0*getRDM(I,J,K,S, rdm );

    else if (K == S && L == R)
      contribution += -1.0*getRDM(I,J,R,S, rdm );

    else if (K == R && R == S)
      contribution += -1.0*getRDM(I,J,L,S, rdm );

    else if (K == R && L == S)
      contribution += 1.0*getRDM(I,J,R,S, rdm );

    else if (R == S)
      contribution += 1.0*getRDM(I,J,R,K,L,S, rdm );

    else if (L == S)
      contribution += -1.0*getRDM(I,J,R,K,R,S, rdm );

    else if (L == R)
      contribution += 1.0*getRDM(I,J,S,K,R,S, rdm );

    else if (K == S)
      contribution += 1.0*getRDM(I,J,R,L,R,S, rdm );

    else if (K == R)
      contribution += -1.0*getRDM(I,J,S,L,R,S, rdm );

    else
      contribution += -1.0*getRDM(I,J,R,S,K,L,R,S, rdm );

    return contribution;
  }
  
  complexT calcTermGJ2f(int P, int Q, int I, int J, int K, int L, int R, MatrixXcT& rdm) {
    complexT contribution = complexT(0.,0.);
    if (P == J && Q == I && L == R)
      contribution += -1.0*getRDM(P,Q,K,R, rdm );

    else if (P == J && Q == I && K == R)
      contribution += 1.0*getRDM(P,Q,L,R, rdm );

    else if (P == I && Q == J && L == R)
      contribution += 1.0*getRDM(P,Q,K,R, rdm );

    else if (P == I && Q == J && K == R)
      contribution += -1.0*getRDM(P,Q,L,R, rdm );

    else if (P == Q && Q == J && L == R)
      contribution += 1.0*getRDM(I,P,K,R, rdm );

    else if (P == Q && Q == J && K == R)
      contribution += -1.0*getRDM(I,P,L,R, rdm );

    else if (P == Q && Q == I && L == R)
      contribution += -1.0*getRDM(J,P,K,R, rdm );

    else if (P == Q && Q == I && K == R)
      contribution += 1.0*getRDM(J,P,L,R, rdm );

    else if (Q == J && L == R)
      contribution += 1.0*getRDM(I,P,Q,K,P,R, rdm );

    else if (Q == J && K == R)
      contribution += -1.0*getRDM(I,P,Q,L,P,R, rdm );

    else if (Q == I && L == R)
      contribution += -1.0*getRDM(J,P,Q,K,P,R, rdm );

    else if (Q == I && K == R)
      contribution += 1.0*getRDM(J,P,Q,L,P,R, rdm );

    else if (P == R && Q == J)
      contribution += -1.0*getRDM(I,P,Q,K,L,R, rdm );

    else if (P == R && Q == I)
      contribution += 1.0*getRDM(J,P,Q,K,L,R, rdm );

    else if (P == J && L == R)
      contribution += -1.0*getRDM(I,P,Q,K,Q,R, rdm );

    else if (P == J && K == R)
      contribution += 1.0*getRDM(I,P,Q,L,Q,R, rdm );

    else if (P == J && Q == R)
      contribution += 1.0*getRDM(I,P,Q,K,L,R, rdm );

    else if (P == J && Q == I)
      contribution += -1.0*getRDM(P,Q,R,K,L,R, rdm );

    else if (P == I && L == R)
      contribution += 1.0*getRDM(J,P,Q,K,Q,R, rdm );

    else if (P == I && K == R)
      contribution += -1.0*getRDM(J,P,Q,L,Q,R, rdm );

    else if (P == I && Q == R)
      contribution += -1.0*getRDM(J,P,Q,K,L,R, rdm );

    else if (P == I && Q == J)
      contribution += 1.0*getRDM(P,Q,R,K,L,R, rdm );

    else if (P == Q && L == R)
      contribution += -1.0*getRDM(I,J,P,K,Q,R, rdm );

    else if (P == Q && K == R)
      contribution += 1.0*getRDM(I,J,P,L,Q,R, rdm );

    else if (P == Q && Q == R)
      contribution += 1.0*getRDM(I,J,P,K,L,R, rdm );

    else if (P == Q && Q == J)
      contribution += 1.0*getRDM(I,P,R,K,L,R, rdm );

    else if (P == Q && Q == I)
      contribution += -1.0*getRDM(J,P,R,K,L,R, rdm );

    else if (L == R)
      contribution += -1.0*getRDM(I,J,P,Q,K,P,Q,R, rdm );

    else if (K == R)
      contribution += 1.0*getRDM(I,J,P,Q,L,P,Q,R, rdm );

    else if (Q == R)
      contribution += -1.0*getRDM(I,J,P,Q,K,L,P,R, rdm );

    else if (Q == J)
      contribution += 1.0*getRDM(I,P,Q,R,K,L,P,R, rdm );

    else if (Q == I)
      contribution += -1.0*getRDM(J,P,Q,R,K,L,P,R, rdm );

    else if (P == R)
      contribution += 1.0*getRDM(I,J,P,Q,K,L,Q,R, rdm );

    else if (P == J)
      contribution += -1.0*getRDM(I,P,Q,R,K,L,Q,R, rdm );

    else if (P == I)
      contribution += 1.0*getRDM(J,P,Q,R,K,L,Q,R, rdm );

    else if (P == Q)
      contribution += -1.0*getRDM(I,J,P,R,K,L,Q,R, rdm );

    else
      contribution += -1.0*getRDM(I,J,P,Q,R,K,L,P,Q,R, rdm );

    return contribution;
  }
  
  complexT calcTermGJ2g(int P, int I, int J, int K, int L, int R, int S, MatrixXcT& rdm) {
    complexT contribution = complexT(0.,0.);

    if (P == J && L == R && R == S)
      contribution += 1.0*getRDM(I,P,K,S, rdm );

    else if (P == J && K == S && L == R)
      contribution += -1.0*getRDM(I,P,R,S, rdm );

    else if (P == J && K == R && R == S)
      contribution += -1.0*getRDM(I,P,L,S, rdm );

    else if (P == J && K == R && L == S)
      contribution += 1.0*getRDM(I,P,R,S, rdm );

    else if (P == I && L == R && R == S)
      contribution += -1.0*getRDM(J,P,K,S, rdm );

    else if (P == I && K == S && L == R)
      contribution += 1.0*getRDM(J,P,R,S, rdm );

    else if (P == I && K == R && R == S)
      contribution += 1.0*getRDM(J,P,L,S, rdm );

    else if (P == I && K == R && L == S)
      contribution += -1.0*getRDM(J,P,R,S, rdm );

    else if (L == R && R == S)
      contribution += -1.0*getRDM(I,J,P,K,P,S, rdm );

    else if (K == S && L == R)
      contribution += -1.0*getRDM(I,J,P,P,R,S, rdm );

    else if (K == R && R == S)
      contribution += 1.0*getRDM(I,J,P,L,P,S, rdm );

    else if (K == R && L == S)
      contribution += 1.0*getRDM(I,J,P,P,R,S, rdm );

    else if (P == S && L == R)
      contribution += 1.0*getRDM(I,J,P,K,R,S, rdm );

    else if (P == S && K == R)
      contribution += -1.0*getRDM(I,J,P,L,R,S, rdm );

    else if (P == R && R == S)
      contribution += 1.0*getRDM(I,J,P,K,L,S, rdm );

    else if (P == R && L == S)
      contribution += -1.0*getRDM(I,J,P,K,R,S, rdm );

    else if (P == R && K == S)
      contribution += 1.0*getRDM(I,J,P,L,R,S, rdm );

    else if (P == J && R == S)
      contribution += 1.0*getRDM(I,P,R,K,L,S, rdm );

    else if (P == J && L == S)
      contribution += -1.0*getRDM(I,P,R,K,R,S, rdm );

    else if (P == J && L == R)
      contribution += 1.0*getRDM(I,P,S,K,R,S, rdm );

    else if (P == J && K == S)
      contribution += 1.0*getRDM(I,P,R,L,R,S, rdm );

    else if (P == J && K == R)
      contribution += -1.0*getRDM(I,P,S,L,R,S, rdm );

    else if (P == I && R == S)
      contribution += -1.0*getRDM(J,P,R,K,L,S, rdm );

    else if (P == I && L == S)
      contribution += 1.0*getRDM(J,P,R,K,R,S, rdm );

    else if (P == I && L == R)
      contribution += -1.0*getRDM(J,P,S,K,R,S, rdm );

    else if (P == I && K == S)
      contribution += -1.0*getRDM(J,P,R,L,R,S, rdm );

    else if (P == I && K == R)
      contribution += 1.0*getRDM(J,P,S,L,R,S, rdm );

    else if (R == S)
      contribution += -1.0*getRDM(I,J,P,R,K,L,P,S, rdm );

    else if (L == S)
      contribution += -1.0*getRDM(I,J,P,R,K,P,R,S, rdm );

    else if (L == R)
      contribution += 1.0*getRDM(I,J,P,S,K,P,R,S, rdm );

    else if (K == S)
      contribution += 1.0*getRDM(I,J,P,R,L,P,R,S, rdm );

    else if (K == R)
      contribution += -1.0*getRDM(I,J,P,S,L,P,R,S, rdm );

    else if (P == S)
      contribution += 1.0*getRDM(I,J,P,R,K,L,R,S, rdm );

    else if (P == R)
      contribution += -1.0*getRDM(I,J,P,S,K,L,R,S, rdm );

    else if (P == J)
      contribution += -1.0*getRDM(I,P,R,S,K,L,R,S, rdm );

    else if (P == I)
      contribution += 1.0*getRDM(J,P,R,S,K,L,R,S, rdm );

    else
      contribution += -1.0*getRDM(I,J,P,R,S,K,L,P,R,S, rdm );

    return contribution;

    
  }
  
  complexT calcTermGJ2h(int P, int Q, int I, int J, int K, int L, int R, int S, MatrixXcT& rdm) {
    complexT contribution = complexT(0.,0.);
    if (P == J && Q == I && L == R && R == S)
      contribution += -1.0*getRDM(P,Q,K,S, rdm );

    else if (P == J && Q == I && K == S && L == R)
      contribution += 1.0*getRDM(P,Q,R,S, rdm );

    else if (P == J && Q == I && K == R && R == S)
      contribution += 1.0*getRDM(P,Q,L,S, rdm );

    else if (P == J && Q == I && K == R && L == S)
      contribution += -1.0*getRDM(P,Q,R,S, rdm );

    else if (P == I && Q == J && L == R && R == S)
      contribution += 1.0*getRDM(P,Q,K,S, rdm );

    else if (P == I && Q == J && K == S && L == R)
      contribution += -1.0*getRDM(P,Q,R,S, rdm );

    else if (P == I && Q == J && K == R && R == S)
      contribution += -1.0*getRDM(P,Q,L,S, rdm );

    else if (P == I && Q == J && K == R && L == S)
      contribution += 1.0*getRDM(P,Q,R,S, rdm );

    else if (P == Q && Q == J && L == R && R == S)
      contribution += 1.0*getRDM(I,P,K,S, rdm );

    else if (P == Q && Q == J && K == S && L == R)
      contribution += -1.0*getRDM(I,P,R,S, rdm );

    else if (P == Q && Q == J && K == R && R == S)
      contribution += -1.0*getRDM(I,P,L,S, rdm );

    else if (P == Q && Q == J && K == R && L == S)
      contribution += 1.0*getRDM(I,P,R,S, rdm );

    else if (P == Q && Q == I && L == R && R == S)
      contribution += -1.0*getRDM(J,P,K,S, rdm );

    else if (P == Q && Q == I && K == S && L == R)
      contribution += 1.0*getRDM(J,P,R,S, rdm );

    else if (P == Q && Q == I && K == R && R == S)
      contribution += 1.0*getRDM(J,P,L,S, rdm );

    else if (P == Q && Q == I && K == R && L == S)
      contribution += -1.0*getRDM(J,P,R,S, rdm );

    else if (Q == J && L == R && R == S)
      contribution += 1.0*getRDM(I,P,Q,K,P,S, rdm );

    else if (Q == J && K == S && L == R)
      contribution += 1.0*getRDM(I,P,Q,P,R,S, rdm );

    else if (Q == J && K == R && R == S)
      contribution += -1.0*getRDM(I,P,Q,L,P,S, rdm );

    else if (Q == J && K == R && L == S)
      contribution += -1.0*getRDM(I,P,Q,P,R,S, rdm );

    else if (Q == I && L == R && R == S)
      contribution += -1.0*getRDM(J,P,Q,K,P,S, rdm );

    else if (Q == I && K == S && L == R)
      contribution += -1.0*getRDM(J,P,Q,P,R,S, rdm );

    else if (Q == I && K == R && R == S)
      contribution += 1.0*getRDM(J,P,Q,L,P,S, rdm );

    else if (Q == I && K == R && L == S)
      contribution += 1.0*getRDM(J,P,Q,P,R,S, rdm );

    else if (P == S && Q == J && L == R)
      contribution += -1.0*getRDM(I,P,Q,K,R,S, rdm );

    else if (P == S && Q == J && K == R)
      contribution += 1.0*getRDM(I,P,Q,L,R,S, rdm );

    else if (P == S && Q == I && L == R)
      contribution += 1.0*getRDM(J,P,Q,K,R,S, rdm );

    else if (P == S && Q == I && K == R)
      contribution += -1.0*getRDM(J,P,Q,L,R,S, rdm );

    else if (P == R && Q == J && R == S)
      contribution += -1.0*getRDM(I,P,Q,K,L,S, rdm );

    else if (P == R && Q == J && L == S)
      contribution += 1.0*getRDM(I,P,Q,K,R,S, rdm );

    else if (P == R && Q == J && K == S)
      contribution += -1.0*getRDM(I,P,Q,L,R,S, rdm );

    else if (P == R && Q == I && R == S)
      contribution += 1.0*getRDM(J,P,Q,K,L,S, rdm );

    else if (P == R && Q == I && L == S)
      contribution += -1.0*getRDM(J,P,Q,K,R,S, rdm );

    else if (P == R && Q == I && K == S)
      contribution += 1.0*getRDM(J,P,Q,L,R,S, rdm );

    else if (P == J && L == R && R == S)
      contribution += -1.0*getRDM(I,P,Q,K,Q,S, rdm );

    else if (P == J && K == S && L == R)
      contribution += -1.0*getRDM(I,P,Q,Q,R,S, rdm );

    else if (P == J && K == R && R == S)
      contribution += 1.0*getRDM(I,P,Q,L,Q,S, rdm );

    else if (P == J && K == R && L == S)
      contribution += 1.0*getRDM(I,P,Q,Q,R,S, rdm );

    else if (P == J && Q == S && L == R)
      contribution += 1.0*getRDM(I,P,Q,K,R,S, rdm );

    else if (P == J && Q == S && K == R)
      contribution += -1.0*getRDM(I,P,Q,L,R,S, rdm );

    else if (P == J && Q == R && R == S)
      contribution += 1.0*getRDM(I,P,Q,K,L,S, rdm );

    else if (P == J && Q == R && L == S)
      contribution += -1.0*getRDM(I,P,Q,K,R,S, rdm );

    else if (P == J && Q == R && K == S)
      contribution += 1.0*getRDM(I,P,Q,L,R,S, rdm );

    else if (P == J && Q == I && R == S)
      contribution += -1.0*getRDM(P,Q,R,K,L,S, rdm );

    else if (P == J && Q == I && L == S)
      contribution += 1.0*getRDM(P,Q,R,K,R,S, rdm );

    else if (P == J && Q == I && L == R)
      contribution += -1.0*getRDM(P,Q,S,K,R,S, rdm );

    else if (P == J && Q == I && K == S)
      contribution += -1.0*getRDM(P,Q,R,L,R,S, rdm );

    else if (P == J && Q == I && K == R)
      contribution += 1.0*getRDM(P,Q,S,L,R,S, rdm );

    else if (P == I && L == R && R == S)
      contribution += 1.0*getRDM(J,P,Q,K,Q,S, rdm );

    else if (P == I && K == S && L == R)
      contribution += 1.0*getRDM(J,P,Q,Q,R,S, rdm );

    else if (P == I && K == R && R == S)
      contribution += -1.0*getRDM(J,P,Q,L,Q,S, rdm );

    else if (P == I && K == R && L == S)
      contribution += -1.0*getRDM(J,P,Q,Q,R,S, rdm );

    else if (P == I && Q == S && L == R)
      contribution += -1.0*getRDM(J,P,Q,K,R,S, rdm );

    else if (P == I && Q == S && K == R)
      contribution += 1.0*getRDM(J,P,Q,L,R,S, rdm );

    else if (P == I && Q == R && R == S)
      contribution += -1.0*getRDM(J,P,Q,K,L,S, rdm );

    else if (P == I && Q == R && L == S)
      contribution += 1.0*getRDM(J,P,Q,K,R,S, rdm );

    else if (P == I && Q == R && K == S)
      contribution += -1.0*getRDM(J,P,Q,L,R,S, rdm );

    else if (P == I && Q == J && R == S)
      contribution += 1.0*getRDM(P,Q,R,K,L,S, rdm );

    else if (P == I && Q == J && L == S)
      contribution += -1.0*getRDM(P,Q,R,K,R,S, rdm );

    else if (P == I && Q == J && L == R)
      contribution += 1.0*getRDM(P,Q,S,K,R,S, rdm );

    else if (P == I && Q == J && K == S)
      contribution += 1.0*getRDM(P,Q,R,L,R,S, rdm );

    else if (P == I && Q == J && K == R)
      contribution += -1.0*getRDM(P,Q,S,L,R,S, rdm );

    else if (P == Q && L == R && R == S)
      contribution += -1.0*getRDM(I,J,P,K,Q,S, rdm );

    else if (P == Q && K == S && L == R)
      contribution += -1.0*getRDM(I,J,P,Q,R,S, rdm );

    else if (P == Q && K == R && R == S)
      contribution += 1.0*getRDM(I,J,P,L,Q,S, rdm );

    else if (P == Q && K == R && L == S)
      contribution += 1.0*getRDM(I,J,P,Q,R,S, rdm );

    else if (P == Q && Q == S && L == R)
      contribution += 1.0*getRDM(I,J,P,K,R,S, rdm );

    else if (P == Q && Q == S && K == R)
      contribution += -1.0*getRDM(I,J,P,L,R,S, rdm );

    else if (P == Q && Q == R && R == S)
      contribution += 1.0*getRDM(I,J,P,K,L,S, rdm );

    else if (P == Q && Q == R && L == S)
      contribution += -1.0*getRDM(I,J,P,K,R,S, rdm );

    else if (P == Q && Q == R && K == S)
      contribution += 1.0*getRDM(I,J,P,L,R,S, rdm );

    else if (P == Q && Q == J && R == S)
      contribution += 1.0*getRDM(I,P,R,K,L,S, rdm );

    else if (P == Q && Q == J && L == S)
      contribution += -1.0*getRDM(I,P,R,K,R,S, rdm );

    else if (P == Q && Q == J && L == R)
      contribution += 1.0*getRDM(I,P,S,K,R,S, rdm );

    else if (P == Q && Q == J && K == S)
      contribution += 1.0*getRDM(I,P,R,L,R,S, rdm );

    else if (P == Q && Q == J && K == R)
      contribution += -1.0*getRDM(I,P,S,L,R,S, rdm );

    else if (P == Q && Q == I && R == S)
      contribution += -1.0*getRDM(J,P,R,K,L,S, rdm );

    else if (P == Q && Q == I && L == S)
      contribution += 1.0*getRDM(J,P,R,K,R,S, rdm );

    else if (P == Q && Q == I && L == R)
      contribution += -1.0*getRDM(J,P,S,K,R,S, rdm );

    else if (P == Q && Q == I && K == S)
      contribution += -1.0*getRDM(J,P,R,L,R,S, rdm );

    else if (P == Q && Q == I && K == R)
      contribution += 1.0*getRDM(J,P,S,L,R,S, rdm );

    else if (L == R && R == S)
      contribution += -1.0*getRDM(I,J,P,Q,K,P,Q,S, rdm );

    else if (K == S && L == R)
      contribution += 1.0*getRDM(I,J,P,Q,P,Q,R,S, rdm );

    else if (K == R && R == S)
      contribution += 1.0*getRDM(I,J,P,Q,L,P,Q,S, rdm );

    else if (K == R && L == S)
      contribution += -1.0*getRDM(I,J,P,Q,P,Q,R,S, rdm );

    else if (Q == S && L == R)
      contribution += 1.0*getRDM(I,J,P,Q,K,P,R,S, rdm );

    else if (Q == S && K == R)
      contribution += -1.0*getRDM(I,J,P,Q,L,P,R,S, rdm );

    else if (Q == R && R == S)
      contribution += -1.0*getRDM(I,J,P,Q,K,L,P,S, rdm );

    else if (Q == R && L == S)
      contribution += -1.0*getRDM(I,J,P,Q,K,P,R,S, rdm );

    else if (Q == R && K == S)
      contribution += 1.0*getRDM(I,J,P,Q,L,P,R,S, rdm );

    else if (Q == J && R == S)
      contribution += 1.0*getRDM(I,P,Q,R,K,L,P,S, rdm );

    else if (Q == J && L == S)
      contribution += 1.0*getRDM(I,P,Q,R,K,P,R,S, rdm );

    else if (Q == J && L == R)
      contribution += -1.0*getRDM(I,P,Q,S,K,P,R,S, rdm );

    else if (Q == J && K == S)
      contribution += -1.0*getRDM(I,P,Q,R,L,P,R,S, rdm );

    else if (Q == J && K == R)
      contribution += 1.0*getRDM(I,P,Q,S,L,P,R,S, rdm );

    else if (Q == I && R == S)
      contribution += -1.0*getRDM(J,P,Q,R,K,L,P,S, rdm );

    else if (Q == I && L == S)
      contribution += -1.0*getRDM(J,P,Q,R,K,P,R,S, rdm );

    else if (Q == I && L == R)
      contribution += 1.0*getRDM(J,P,Q,S,K,P,R,S, rdm );

    else if (Q == I && K == S)
      contribution += 1.0*getRDM(J,P,Q,R,L,P,R,S, rdm );

    else if (Q == I && K == R)
      contribution += -1.0*getRDM(J,P,Q,S,L,P,R,S, rdm );

    else if (P == S && L == R)
      contribution += -1.0*getRDM(I,J,P,Q,K,Q,R,S, rdm );

    else if (P == S && K == R)
      contribution += 1.0*getRDM(I,J,P,Q,L,Q,R,S, rdm );

    else if (P == S && Q == R)
      contribution += 1.0*getRDM(I,J,P,Q,K,L,R,S, rdm );

    else if (P == S && Q == J)
      contribution += -1.0*getRDM(I,P,Q,R,K,L,R,S, rdm );

    else if (P == S && Q == I)
      contribution += 1.0*getRDM(J,P,Q,R,K,L,R,S, rdm );

    else if (P == R && R == S)
      contribution += 1.0*getRDM(I,J,P,Q,K,L,Q,S, rdm );

    else if (P == R && L == S)
      contribution += 1.0*getRDM(I,J,P,Q,K,Q,R,S, rdm );

    else if (P == R && K == S)
      contribution += -1.0*getRDM(I,J,P,Q,L,Q,R,S, rdm );

    else if (P == R && Q == S)
      contribution += -1.0*getRDM(I,J,P,Q,K,L,R,S, rdm );

    else if (P == R && Q == J)
      contribution += 1.0*getRDM(I,P,Q,S,K,L,R,S, rdm );

    else if (P == R && Q == I)
      contribution += -1.0*getRDM(J,P,Q,S,K,L,R,S, rdm );

    else if (P == J && R == S)
      contribution += -1.0*getRDM(I,P,Q,R,K,L,Q,S, rdm );

    else if (P == J && L == S)
      contribution += -1.0*getRDM(I,P,Q,R,K,Q,R,S, rdm );

    else if (P == J && L == R)
      contribution += 1.0*getRDM(I,P,Q,S,K,Q,R,S, rdm );

    else if (P == J && K == S)
      contribution += 1.0*getRDM(I,P,Q,R,L,Q,R,S, rdm );

    else if (P == J && K == R)
      contribution += -1.0*getRDM(I,P,Q,S,L,Q,R,S, rdm );

    else if (P == J && Q == S)
      contribution += 1.0*getRDM(I,P,Q,R,K,L,R,S, rdm );

    else if (P == J && Q == R)
      contribution += -1.0*getRDM(I,P,Q,S,K,L,R,S, rdm );

    else if (P == J && Q == I)
      contribution += 1.0*getRDM(P,Q,R,S,K,L,R,S, rdm );

    else if (P == I && R == S)
      contribution += 1.0*getRDM(J,P,Q,R,K,L,Q,S, rdm );

    else if (P == I && L == S)
      contribution += 1.0*getRDM(J,P,Q,R,K,Q,R,S, rdm );

    else if (P == I && L == R)
      contribution += -1.0*getRDM(J,P,Q,S,K,Q,R,S, rdm );

    else if (P == I && K == S)
      contribution += -1.0*getRDM(J,P,Q,R,L,Q,R,S, rdm );

    else if (P == I && K == R)
      contribution += 1.0*getRDM(J,P,Q,S,L,Q,R,S, rdm );

    else if (P == I && Q == S)
      contribution += -1.0*getRDM(J,P,Q,R,K,L,R,S, rdm );

    else if (P == I && Q == R)
      contribution += 1.0*getRDM(J,P,Q,S,K,L,R,S, rdm );

    else if (P == I && Q == J)
      contribution += -1.0*getRDM(P,Q,R,S,K,L,R,S, rdm );

    else if (P == Q && R == S)
      contribution += -1.0*getRDM(I,J,P,R,K,L,Q,S, rdm );

    else if (P == Q && L == S)
      contribution += -1.0*getRDM(I,J,P,R,K,Q,R,S, rdm );

    else if (P == Q && L == R)
      contribution += 1.0*getRDM(I,J,P,S,K,Q,R,S, rdm );

    else if (P == Q && K == S)
      contribution += 1.0*getRDM(I,J,P,R,L,Q,R,S, rdm );

    else if (P == Q && K == R)
      contribution += -1.0*getRDM(I,J,P,S,L,Q,R,S, rdm );

    else if (P == Q && Q == S)
      contribution += 1.0*getRDM(I,J,P,R,K,L,R,S, rdm );

    else if (P == Q && Q == R)
      contribution += -1.0*getRDM(I,J,P,S,K,L,R,S, rdm );

    else if (P == Q && Q == J)
      contribution += -1.0*getRDM(I,P,R,S,K,L,R,S, rdm );

    else if (P == Q && Q == I)
      contribution += 1.0*getRDM(J,P,R,S,K,L,R,S, rdm );

    else if (R == S)
      contribution += -1.0*getRDM(I,J,P,Q,R,K,L,P,Q,S, rdm );

    else if (L == S)
      contribution += 1.0*getRDM(I,J,P,Q,R,K,P,Q,R,S, rdm );

    else if (L == R)
      contribution += -1.0*getRDM(I,J,P,Q,S,K,P,Q,R,S, rdm );

    else if (K == S)
      contribution += -1.0*getRDM(I,J,P,Q,R,L,P,Q,R,S, rdm );

    else if (K == R)
      contribution += 1.0*getRDM(I,J,P,Q,S,L,P,Q,R,S, rdm );

    else if (Q == S)
      contribution += 1.0*getRDM(I,J,P,Q,R,K,L,P,R,S, rdm );

    else if (Q == R)
      contribution += -1.0*getRDM(I,J,P,Q,S,K,L,P,R,S, rdm );

    else if (Q == J)
      contribution += 1.0*getRDM(I,P,Q,R,S,K,L,P,R,S, rdm );

    else if (Q == I)
      contribution += -1.0*getRDM(J,P,Q,R,S,K,L,P,R,S, rdm );

    else if (P == S)
      contribution += -1.0*getRDM(I,J,P,Q,R,K,L,Q,R,S, rdm );

    else if (P == R)
      contribution += 1.0*getRDM(I,J,P,Q,S,K,L,Q,R,S, rdm );

    else if (P == J)
      contribution += -1.0*getRDM(I,P,Q,R,S,K,L,Q,R,S, rdm );

    else if (P == I)
      contribution += 1.0*getRDM(J,P,Q,R,S,K,L,Q,R,S, rdm );

    else if (P == Q)
      contribution += -1.0*getRDM(I,J,P,R,S,K,L,Q,R,S, rdm );

    else
      contribution += 1.0*getRDM(I,J,P,Q,R,S,K,L,P,Q,R,S, rdm );

    return contribution;
  }
  
  complexT calcTermGJ2i(int M, int N, int P, int I, int J, int K, int L, MatrixXcT& rdm) {
    complexT contribution = complexT(0.,0.);
    if (M == J && N == P && P == I)
      contribution += -1.0*getRDM(M,N,K,L, rdm );

    else if (M == I && N == P && P == J)
      contribution += 1.0*getRDM(M,N,K,L, rdm );

    else if (M == P && N == J && P == I)
      contribution += 1.0*getRDM(M,N,K,L, rdm );

    else if (M == P && N == I && P == J)
      contribution += -1.0*getRDM(M,N,K,L, rdm );

    else if (M == N && N == J && P == I)
      contribution += -1.0*getRDM(M,P,K,L, rdm );

    else if (M == N && N == I && P == J)
      contribution += 1.0*getRDM(M,P,K,L, rdm );

    else if (M == N && N == P && P == J)
      contribution += 1.0*getRDM(I,M,K,L, rdm );

    else if (M == N && N == P && P == I)
      contribution += -1.0*getRDM(J,M,K,L, rdm );

    else if (N == J && P == I)
      contribution += -1.0*getRDM(M,N,P,K,L,M, rdm );

    else if (N == I && P == J)
      contribution += 1.0*getRDM(M,N,P,K,L,M, rdm );

    else if (N == P && P == J)
      contribution += -1.0*getRDM(I,M,N,K,L,M, rdm );

    else if (N == P && P == I)
      contribution += 1.0*getRDM(J,M,N,K,L,M, rdm );

    else if (M == J && P == I)
      contribution += 1.0*getRDM(M,N,P,K,L,N, rdm );

    else if (M == J && N == I)
      contribution += -1.0*getRDM(M,N,P,K,L,P, rdm );

    else if (M == J && N == P)
      contribution += 1.0*getRDM(I,M,N,K,L,P, rdm );

    else if (M == I && P == J)
      contribution += -1.0*getRDM(M,N,P,K,L,N, rdm );

    else if (M == I && N == J)
      contribution += 1.0*getRDM(M,N,P,K,L,P, rdm );

    else if (M == I && N == P)
      contribution += -1.0*getRDM(J,M,N,K,L,P, rdm );

    else if (M == P && P == J)
      contribution += 1.0*getRDM(I,M,N,K,L,N, rdm );

    else if (M == P && P == I)
      contribution += -1.0*getRDM(J,M,N,K,L,N, rdm );

    else if (M == P && N == J)
      contribution += -1.0*getRDM(I,M,N,K,L,P, rdm );

    else if (M == P && N == I)
      contribution += 1.0*getRDM(J,M,N,K,L,P, rdm );

    else if (M == N && P == J)
      contribution += -1.0*getRDM(I,M,P,K,L,N, rdm );

    else if (M == N && P == I)
      contribution += 1.0*getRDM(J,M,P,K,L,N, rdm );

    else if (M == N && N == J)
      contribution += 1.0*getRDM(I,M,P,K,L,P, rdm );

    else if (M == N && N == I)
      contribution += -1.0*getRDM(J,M,P,K,L,P, rdm );

    else if (M == N && N == P)
      contribution += 1.0*getRDM(I,J,M,K,L,P, rdm );

    else if (P == J)
      contribution += -1.0*getRDM(I,M,N,P,K,L,M,N, rdm );

    else if (P == I)
      contribution += 1.0*getRDM(J,M,N,P,K,L,M,N, rdm );

    else if (N == J)
      contribution += 1.0*getRDM(I,M,N,P,K,L,M,P, rdm );

    else if (N == I)
      contribution += -1.0*getRDM(J,M,N,P,K,L,M,P, rdm );

    else if (N == P)
      contribution += -1.0*getRDM(I,J,M,N,K,L,M,P, rdm );

    else if (M == J)
      contribution += -1.0*getRDM(I,M,N,P,K,L,N,P, rdm );

    else if (M == I)
      contribution += 1.0*getRDM(J,M,N,P,K,L,N,P, rdm );

    else if (M == P)
      contribution += 1.0*getRDM(I,J,M,N,K,L,N,P, rdm );

    else if (M == N)
      contribution += -1.0*getRDM(I,J,M,P,K,L,N,P, rdm );

    else
      contribution += -1.0*getRDM(I,J,M,N,P,K,L,M,N,P, rdm );

    return contribution;
    
  }
  
  complexT calcTermGJ2j(int M, int N, int P, int Q, int I, int J, int K, int L, MatrixXcT& rdm) {
    complexT contribution = complexT(0.,0.);
    if (M == J && N == P && P == Q && Q == I)
      contribution += -1.0*getRDM(M,N,K,L, rdm );

    else if (M == I && N == P && P == Q && Q == J)
      contribution += 1.0*getRDM(M,N,K,L, rdm );

    else if (M == Q && N == P && P == J && Q == I)
      contribution += 1.0*getRDM(M,N,K,L, rdm );

    else if (M == Q && N == P && P == I && Q == J)
      contribution += -1.0*getRDM(M,N,K,L, rdm );

    else if (M == P && N == J && P == Q && Q == I)
      contribution += 1.0*getRDM(M,N,K,L, rdm );

    else if (M == P && N == I && P == Q && Q == J)
      contribution += -1.0*getRDM(M,N,K,L, rdm );

    else if (M == P && N == Q && P == J && Q == I)
      contribution += -1.0*getRDM(M,N,K,L, rdm );

    else if (M == P && N == Q && P == I && Q == J)
      contribution += 1.0*getRDM(M,N,K,L, rdm );

    else if (M == N && N == J && P == Q && Q == I)
      contribution += -1.0*getRDM(M,P,K,L, rdm );

    else if (M == N && N == I && P == Q && Q == J)
      contribution += 1.0*getRDM(M,P,K,L, rdm );

    else if (M == N && N == Q && P == J && Q == I)
      contribution += 1.0*getRDM(M,P,K,L, rdm );

    else if (M == N && N == Q && P == I && Q == J)
      contribution += -1.0*getRDM(M,P,K,L, rdm );

    else if (M == N && N == P && P == J && Q == I)
      contribution += -1.0*getRDM(M,Q,K,L, rdm );

    else if (M == N && N == P && P == I && Q == J)
      contribution += 1.0*getRDM(M,Q,K,L, rdm );

    else if (M == N && N == P && P == Q && Q == J)
      contribution += 1.0*getRDM(I,M,K,L, rdm );

    else if (M == N && N == P && P == Q && Q == I)
      contribution += -1.0*getRDM(J,M,K,L, rdm );

    else if (N == J && P == Q && Q == I)
      contribution += -1.0*getRDM(M,N,P,K,L,M, rdm );

    else if (N == I && P == Q && Q == J)
      contribution += 1.0*getRDM(M,N,P,K,L,M, rdm );

    else if (N == Q && P == J && Q == I)
      contribution += 1.0*getRDM(M,N,P,K,L,M, rdm );

    else if (N == Q && P == I && Q == J)
      contribution += -1.0*getRDM(M,N,P,K,L,M, rdm );

    else if (N == P && P == J && Q == I)
      contribution += -1.0*getRDM(M,N,Q,K,L,M, rdm );

    else if (N == P && P == I && Q == J)
      contribution += 1.0*getRDM(M,N,Q,K,L,M, rdm );

    else if (N == P && P == Q && Q == J)
      contribution += -1.0*getRDM(I,M,N,K,L,M, rdm );

    else if (N == P && P == Q && Q == I)
      contribution += 1.0*getRDM(J,M,N,K,L,M, rdm );

    else if (M == J && P == Q && Q == I)
      contribution += 1.0*getRDM(M,N,P,K,L,N, rdm );

    else if (M == J && N == I && P == Q)
      contribution += -1.0*getRDM(M,N,P,K,L,Q, rdm );

    else if (M == J && N == Q && Q == I)
      contribution += -1.0*getRDM(M,N,P,K,L,P, rdm );

    else if (M == J && N == Q && P == I)
      contribution += 1.0*getRDM(M,N,P,K,L,Q, rdm );

    else if (M == J && N == P && Q == I)
      contribution += 1.0*getRDM(M,N,Q,K,L,P, rdm );

    else if (M == J && N == P && P == I)
      contribution += -1.0*getRDM(M,N,Q,K,L,Q, rdm );

    else if (M == J && N == P && P == Q)
      contribution += 1.0*getRDM(I,M,N,K,L,Q, rdm );

    else if (M == I && P == Q && Q == J)
      contribution += -1.0*getRDM(M,N,P,K,L,N, rdm );

    else if (M == I && N == J && P == Q)
      contribution += 1.0*getRDM(M,N,P,K,L,Q, rdm );

    else if (M == I && N == Q && Q == J)
      contribution += 1.0*getRDM(M,N,P,K,L,P, rdm );

    else if (M == I && N == Q && P == J)
      contribution += -1.0*getRDM(M,N,P,K,L,Q, rdm );

    else if (M == I && N == P && Q == J)
      contribution += -1.0*getRDM(M,N,Q,K,L,P, rdm );

    else if (M == I && N == P && P == J)
      contribution += 1.0*getRDM(M,N,Q,K,L,Q, rdm );

    else if (M == I && N == P && P == Q)
      contribution += -1.0*getRDM(J,M,N,K,L,Q, rdm );

    else if (M == Q && P == J && Q == I)
      contribution += -1.0*getRDM(M,N,P,K,L,N, rdm );

    else if (M == Q && P == I && Q == J)
      contribution += 1.0*getRDM(M,N,P,K,L,N, rdm );

    else if (M == Q && N == J && Q == I)
      contribution += 1.0*getRDM(M,N,P,K,L,P, rdm );

    else if (M == Q && N == J && P == I)
      contribution += -1.0*getRDM(M,N,P,K,L,Q, rdm );

    else if (M == Q && N == I && Q == J)
      contribution += -1.0*getRDM(M,N,P,K,L,P, rdm );

    else if (M == Q && N == I && P == J)
      contribution += 1.0*getRDM(M,N,P,K,L,Q, rdm );

    else if (M == Q && N == P && Q == J)
      contribution += 1.0*getRDM(I,M,N,K,L,P, rdm );

    else if (M == Q && N == P && Q == I)
      contribution += -1.0*getRDM(J,M,N,K,L,P, rdm );

    else if (M == Q && N == P && P == J)
      contribution += -1.0*getRDM(I,M,N,K,L,Q, rdm );

    else if (M == Q && N == P && P == I)
      contribution += 1.0*getRDM(J,M,N,K,L,Q, rdm );

    else if (M == P && P == J && Q == I)
      contribution += 1.0*getRDM(M,N,Q,K,L,N, rdm );

    else if (M == P && P == I && Q == J)
      contribution += -1.0*getRDM(M,N,Q,K,L,N, rdm );

    else if (M == P && P == Q && Q == J)
      contribution += 1.0*getRDM(I,M,N,K,L,N, rdm );

    else if (M == P && P == Q && Q == I)
      contribution += -1.0*getRDM(J,M,N,K,L,N, rdm );

    else if (M == P && N == J && Q == I)
      contribution += -1.0*getRDM(M,N,Q,K,L,P, rdm );

    else if (M == P && N == J && P == I)
      contribution += 1.0*getRDM(M,N,Q,K,L,Q, rdm );

    else if (M == P && N == J && P == Q)
      contribution += -1.0*getRDM(I,M,N,K,L,Q, rdm );

    else if (M == P && N == I && Q == J)
      contribution += 1.0*getRDM(M,N,Q,K,L,P, rdm );

    else if (M == P && N == I && P == J)
      contribution += -1.0*getRDM(M,N,Q,K,L,Q, rdm );

    else if (M == P && N == I && P == Q)
      contribution += 1.0*getRDM(J,M,N,K,L,Q, rdm );

    else if (M == P && N == Q && Q == J)
      contribution += -1.0*getRDM(I,M,N,K,L,P, rdm );

    else if (M == P && N == Q && Q == I)
      contribution += 1.0*getRDM(J,M,N,K,L,P, rdm );

    else if (M == P && N == Q && P == J)
      contribution += 1.0*getRDM(I,M,N,K,L,Q, rdm );

    else if (M == P && N == Q && P == I)
      contribution += -1.0*getRDM(J,M,N,K,L,Q, rdm );

    else if (M == N && P == J && Q == I)
      contribution += -1.0*getRDM(M,P,Q,K,L,N, rdm );

    else if (M == N && P == I && Q == J)
      contribution += 1.0*getRDM(M,P,Q,K,L,N, rdm );

    else if (M == N && P == Q && Q == J)
      contribution += -1.0*getRDM(I,M,P,K,L,N, rdm );

    else if (M == N && P == Q && Q == I)
      contribution += 1.0*getRDM(J,M,P,K,L,N, rdm );

    else if (M == N && N == J && Q == I)
      contribution += 1.0*getRDM(M,P,Q,K,L,P, rdm );

    else if (M == N && N == J && P == I)
      contribution += -1.0*getRDM(M,P,Q,K,L,Q, rdm );

    else if (M == N && N == J && P == Q)
      contribution += 1.0*getRDM(I,M,P,K,L,Q, rdm );

    else if (M == N && N == I && Q == J)
      contribution += -1.0*getRDM(M,P,Q,K,L,P, rdm );

    else if (M == N && N == I && P == J)
      contribution += 1.0*getRDM(M,P,Q,K,L,Q, rdm );

    else if (M == N && N == I && P == Q)
      contribution += -1.0*getRDM(J,M,P,K,L,Q, rdm );

    else if (M == N && N == Q && Q == J)
      contribution += 1.0*getRDM(I,M,P,K,L,P, rdm );

    else if (M == N && N == Q && Q == I)
      contribution += -1.0*getRDM(J,M,P,K,L,P, rdm );

    else if (M == N && N == Q && P == J)
      contribution += -1.0*getRDM(I,M,P,K,L,Q, rdm );

    else if (M == N && N == Q && P == I)
      contribution += 1.0*getRDM(J,M,P,K,L,Q, rdm );

    else if (M == N && N == P && Q == J)
      contribution += -1.0*getRDM(I,M,Q,K,L,P, rdm );

    else if (M == N && N == P && Q == I)
      contribution += 1.0*getRDM(J,M,Q,K,L,P, rdm );

    else if (M == N && N == P && P == J)
      contribution += 1.0*getRDM(I,M,Q,K,L,Q, rdm );

    else if (M == N && N == P && P == I)
      contribution += -1.0*getRDM(J,M,Q,K,L,Q, rdm );

    else if (M == N && N == P && P == Q)
      contribution += 1.0*getRDM(I,J,M,K,L,Q, rdm );

    else if (P == J && Q == I)
      contribution += 1.0*getRDM(M,N,P,Q,K,L,M,N, rdm );

    else if (P == I && Q == J)
      contribution += -1.0*getRDM(M,N,P,Q,K,L,M,N, rdm );

    else if (P == Q && Q == J)
      contribution += -1.0*getRDM(I,M,N,P,K,L,M,N, rdm );

    else if (P == Q && Q == I)
      contribution += 1.0*getRDM(J,M,N,P,K,L,M,N, rdm );

    else if (N == J && Q == I)
      contribution += -1.0*getRDM(M,N,P,Q,K,L,M,P, rdm );

    else if (N == J && P == I)
      contribution += 1.0*getRDM(M,N,P,Q,K,L,M,Q, rdm );

    else if (N == J && P == Q)
      contribution += 1.0*getRDM(I,M,N,P,K,L,M,Q, rdm );

    else if (N == I && Q == J)
      contribution += 1.0*getRDM(M,N,P,Q,K,L,M,P, rdm );

    else if (N == I && P == J)
      contribution += -1.0*getRDM(M,N,P,Q,K,L,M,Q, rdm );

    else if (N == I && P == Q)
      contribution += -1.0*getRDM(J,M,N,P,K,L,M,Q, rdm );

    else if (N == Q && Q == J)
      contribution += 1.0*getRDM(I,M,N,P,K,L,M,P, rdm );

    else if (N == Q && Q == I)
      contribution += -1.0*getRDM(J,M,N,P,K,L,M,P, rdm );

    else if (N == Q && P == J)
      contribution += -1.0*getRDM(I,M,N,P,K,L,M,Q, rdm );

    else if (N == Q && P == I)
      contribution += 1.0*getRDM(J,M,N,P,K,L,M,Q, rdm );

    else if (N == P && Q == J)
      contribution += -1.0*getRDM(I,M,N,Q,K,L,M,P, rdm );

    else if (N == P && Q == I)
      contribution += 1.0*getRDM(J,M,N,Q,K,L,M,P, rdm );

    else if (N == P && P == J)
      contribution += 1.0*getRDM(I,M,N,Q,K,L,M,Q, rdm );

    else if (N == P && P == I)
      contribution += -1.0*getRDM(J,M,N,Q,K,L,M,Q, rdm );

    else if (N == P && P == Q)
      contribution += -1.0*getRDM(I,J,M,N,K,L,M,Q, rdm );

    else if (M == J && Q == I)
      contribution += 1.0*getRDM(M,N,P,Q,K,L,N,P, rdm );

    else if (M == J && P == I)
      contribution += -1.0*getRDM(M,N,P,Q,K,L,N,Q, rdm );

    else if (M == J && P == Q)
      contribution += -1.0*getRDM(I,M,N,P,K,L,N,Q, rdm );

    else if (M == J && N == I)
      contribution += 1.0*getRDM(M,N,P,Q,K,L,P,Q, rdm );

    else if (M == J && N == Q)
      contribution += 1.0*getRDM(I,M,N,P,K,L,P,Q, rdm );

    else if (M == J && N == P)
      contribution += -1.0*getRDM(I,M,N,Q,K,L,P,Q, rdm );

    else if (M == I && Q == J)
      contribution += -1.0*getRDM(M,N,P,Q,K,L,N,P, rdm );

    else if (M == I && P == J)
      contribution += 1.0*getRDM(M,N,P,Q,K,L,N,Q, rdm );

    else if (M == I && P == Q)
      contribution += 1.0*getRDM(J,M,N,P,K,L,N,Q, rdm );

    else if (M == I && N == J)
      contribution += -1.0*getRDM(M,N,P,Q,K,L,P,Q, rdm );

    else if (M == I && N == Q)
      contribution += -1.0*getRDM(J,M,N,P,K,L,P,Q, rdm );

    else if (M == I && N == P)
      contribution += 1.0*getRDM(J,M,N,Q,K,L,P,Q, rdm );

    else if (M == Q && Q == J)
      contribution += -1.0*getRDM(I,M,N,P,K,L,N,P, rdm );

    else if (M == Q && Q == I)
      contribution += 1.0*getRDM(J,M,N,P,K,L,N,P, rdm );

    else if (M == Q && P == J)
      contribution += 1.0*getRDM(I,M,N,P,K,L,N,Q, rdm );

    else if (M == Q && P == I)
      contribution += -1.0*getRDM(J,M,N,P,K,L,N,Q, rdm );

    else if (M == Q && N == J)
      contribution += -1.0*getRDM(I,M,N,P,K,L,P,Q, rdm );

    else if (M == Q && N == I)
      contribution += 1.0*getRDM(J,M,N,P,K,L,P,Q, rdm );

    else if (M == Q && N == P)
      contribution += 1.0*getRDM(I,J,M,N,K,L,P,Q, rdm );

    else if (M == P && Q == J)
      contribution += 1.0*getRDM(I,M,N,Q,K,L,N,P, rdm );

    else if (M == P && Q == I)
      contribution += -1.0*getRDM(J,M,N,Q,K,L,N,P, rdm );

    else if (M == P && P == J)
      contribution += -1.0*getRDM(I,M,N,Q,K,L,N,Q, rdm );

    else if (M == P && P == I)
      contribution += 1.0*getRDM(J,M,N,Q,K,L,N,Q, rdm );

    else if (M == P && P == Q)
      contribution += 1.0*getRDM(I,J,M,N,K,L,N,Q, rdm );

    else if (M == P && N == J)
      contribution += 1.0*getRDM(I,M,N,Q,K,L,P,Q, rdm );

    else if (M == P && N == I)
      contribution += -1.0*getRDM(J,M,N,Q,K,L,P,Q, rdm );

    else if (M == P && N == Q)
      contribution += -1.0*getRDM(I,J,M,N,K,L,P,Q, rdm );

    else if (M == N && Q == J)
      contribution += -1.0*getRDM(I,M,P,Q,K,L,N,P, rdm );

    else if (M == N && Q == I)
      contribution += 1.0*getRDM(J,M,P,Q,K,L,N,P, rdm );

    else if (M == N && P == J)
      contribution += 1.0*getRDM(I,M,P,Q,K,L,N,Q, rdm );

    else if (M == N && P == I)
      contribution += -1.0*getRDM(J,M,P,Q,K,L,N,Q, rdm );

    else if (M == N && P == Q)
      contribution += -1.0*getRDM(I,J,M,P,K,L,N,Q, rdm );

    else if (M == N && N == J)
      contribution += -1.0*getRDM(I,M,P,Q,K,L,P,Q, rdm );

    else if (M == N && N == I)
      contribution += 1.0*getRDM(J,M,P,Q,K,L,P,Q, rdm );

    else if (M == N && N == Q)
      contribution += 1.0*getRDM(I,J,M,P,K,L,P,Q, rdm );

    else if (M == N && N == P)
      contribution += -1.0*getRDM(I,J,M,Q,K,L,P,Q, rdm );

    else if (Q == J)
      contribution += 1.0*getRDM(I,M,N,P,Q,K,L,M,N,P, rdm );

    else if (Q == I)
      contribution += -1.0*getRDM(J,M,N,P,Q,K,L,M,N,P, rdm );

    else if (P == J)
      contribution += -1.0*getRDM(I,M,N,P,Q,K,L,M,N,Q, rdm );

    else if (P == I)
      contribution += 1.0*getRDM(J,M,N,P,Q,K,L,M,N,Q, rdm );

    else if (P == Q)
      contribution += -1.0*getRDM(I,J,M,N,P,K,L,M,N,Q, rdm );

    else if (N == J)
      contribution += 1.0*getRDM(I,M,N,P,Q,K,L,M,P,Q, rdm );

    else if (N == I)
      contribution += -1.0*getRDM(J,M,N,P,Q,K,L,M,P,Q, rdm );

    else if (N == Q)
      contribution += 1.0*getRDM(I,J,M,N,P,K,L,M,P,Q, rdm );

    else if (N == P)
      contribution += -1.0*getRDM(I,J,M,N,Q,K,L,M,P,Q, rdm );

    else if (M == J)
      contribution += -1.0*getRDM(I,M,N,P,Q,K,L,N,P,Q, rdm );

    else if (M == I)
      contribution += 1.0*getRDM(J,M,N,P,Q,K,L,N,P,Q, rdm );

    else if (M == Q)
      contribution += -1.0*getRDM(I,J,M,N,P,K,L,N,P,Q, rdm );

    else if (M == P)
      contribution += 1.0*getRDM(I,J,M,N,Q,K,L,N,P,Q, rdm );

    else if (M == N)
      contribution += -1.0*getRDM(I,J,M,P,Q,K,L,N,P,Q, rdm );

    else
      contribution += 1.0*getRDM(I,J,M,N,P,Q,K,L,M,N,P,Q, rdm );

    return contribution;
    
  }
  
  complexT calcTermGJ2k(int M, int N, int P, int I, int J, int K, int L, int R, MatrixXcT& rdm) {
    complexT contribution = complexT(0.,0.);
    if (M == J && N == P && P == I && L == R)
      contribution += -1.0*getRDM(M,N,K,R, rdm );

    else if (M == J && N == P && P == I && K == R)
      contribution += 1.0*getRDM(M,N,L,R, rdm );

    else if (M == I && N == P && P == J && L == R)
      contribution += 1.0*getRDM(M,N,K,R, rdm );

    else if (M == I && N == P && P == J && K == R)
      contribution += -1.0*getRDM(M,N,L,R, rdm );

    else if (M == P && N == J && P == I && L == R)
      contribution += 1.0*getRDM(M,N,K,R, rdm );

    else if (M == P && N == J && P == I && K == R)
      contribution += -1.0*getRDM(M,N,L,R, rdm );

    else if (M == P && N == I && P == J && L == R)
      contribution += -1.0*getRDM(M,N,K,R, rdm );

    else if (M == P && N == I && P == J && K == R)
      contribution += 1.0*getRDM(M,N,L,R, rdm );

    else if (M == N && N == J && P == I && L == R)
      contribution += -1.0*getRDM(M,P,K,R, rdm );

    else if (M == N && N == J && P == I && K == R)
      contribution += 1.0*getRDM(M,P,L,R, rdm );

    else if (M == N && N == I && P == J && L == R)
      contribution += 1.0*getRDM(M,P,K,R, rdm );

    else if (M == N && N == I && P == J && K == R)
      contribution += -1.0*getRDM(M,P,L,R, rdm );

    else if (M == N && N == P && P == J && L == R)
      contribution += 1.0*getRDM(I,M,K,R, rdm );

    else if (M == N && N == P && P == J && K == R)
      contribution += -1.0*getRDM(I,M,L,R, rdm );

    else if (M == N && N == P && P == I && L == R)
      contribution += -1.0*getRDM(J,M,K,R, rdm );

    else if (M == N && N == P && P == I && K == R)
      contribution += 1.0*getRDM(J,M,L,R, rdm );

    else if (N == J && P == I && L == R)
      contribution += 1.0*getRDM(M,N,P,K,M,R, rdm );

    else if (N == J && P == I && K == R)
      contribution += -1.0*getRDM(M,N,P,L,M,R, rdm );

    else if (N == I && P == J && L == R)
      contribution += -1.0*getRDM(M,N,P,K,M,R, rdm );

    else if (N == I && P == J && K == R)
      contribution += 1.0*getRDM(M,N,P,L,M,R, rdm );

    else if (N == P && P == J && L == R)
      contribution += 1.0*getRDM(I,M,N,K,M,R, rdm );

    else if (N == P && P == J && K == R)
      contribution += -1.0*getRDM(I,M,N,L,M,R, rdm );

    else if (N == P && P == I && L == R)
      contribution += -1.0*getRDM(J,M,N,K,M,R, rdm );

    else if (N == P && P == I && K == R)
      contribution += 1.0*getRDM(J,M,N,L,M,R, rdm );

    else if (M == R && N == J && P == I)
      contribution += -1.0*getRDM(M,N,P,K,L,R, rdm );

    else if (M == R && N == I && P == J)
      contribution += 1.0*getRDM(M,N,P,K,L,R, rdm );

    else if (M == R && N == P && P == J)
      contribution += -1.0*getRDM(I,M,N,K,L,R, rdm );

    else if (M == R && N == P && P == I)
      contribution += 1.0*getRDM(J,M,N,K,L,R, rdm );

    else if (M == J && P == I && L == R)
      contribution += -1.0*getRDM(M,N,P,K,N,R, rdm );

    else if (M == J && P == I && K == R)
      contribution += 1.0*getRDM(M,N,P,L,N,R, rdm );

    else if (M == J && N == R && P == I)
      contribution += 1.0*getRDM(M,N,P,K,L,R, rdm );

    else if (M == J && N == I && L == R)
      contribution += 1.0*getRDM(M,N,P,K,P,R, rdm );

    else if (M == J && N == I && K == R)
      contribution += -1.0*getRDM(M,N,P,L,P,R, rdm );

    else if (M == J && N == I && P == R)
      contribution += -1.0*getRDM(M,N,P,K,L,R, rdm );

    else if (M == J && N == P && L == R)
      contribution += -1.0*getRDM(I,M,N,K,P,R, rdm );

    else if (M == J && N == P && K == R)
      contribution += 1.0*getRDM(I,M,N,L,P,R, rdm );

    else if (M == J && N == P && P == R)
      contribution += 1.0*getRDM(I,M,N,K,L,R, rdm );

    else if (M == J && N == P && P == I)
      contribution += -1.0*getRDM(M,N,R,K,L,R, rdm );

    else if (M == I && P == J && L == R)
      contribution += 1.0*getRDM(M,N,P,K,N,R, rdm );

    else if (M == I && P == J && K == R)
      contribution += -1.0*getRDM(M,N,P,L,N,R, rdm );

    else if (M == I && N == R && P == J)
      contribution += -1.0*getRDM(M,N,P,K,L,R, rdm );

    else if (M == I && N == J && L == R)
      contribution += -1.0*getRDM(M,N,P,K,P,R, rdm );

    else if (M == I && N == J && K == R)
      contribution += 1.0*getRDM(M,N,P,L,P,R, rdm );

    else if (M == I && N == J && P == R)
      contribution += 1.0*getRDM(M,N,P,K,L,R, rdm );

    else if (M == I && N == P && L == R)
      contribution += 1.0*getRDM(J,M,N,K,P,R, rdm );

    else if (M == I && N == P && K == R)
      contribution += -1.0*getRDM(J,M,N,L,P,R, rdm );

    else if (M == I && N == P && P == R)
      contribution += -1.0*getRDM(J,M,N,K,L,R, rdm );

    else if (M == I && N == P && P == J)
      contribution += 1.0*getRDM(M,N,R,K,L,R, rdm );

    else if (M == P && P == J && L == R)
      contribution += -1.0*getRDM(I,M,N,K,N,R, rdm );

    else if (M == P && P == J && K == R)
      contribution += 1.0*getRDM(I,M,N,L,N,R, rdm );

    else if (M == P && P == I && L == R)
      contribution += 1.0*getRDM(J,M,N,K,N,R, rdm );

    else if (M == P && P == I && K == R)
      contribution += -1.0*getRDM(J,M,N,L,N,R, rdm );

    else if (M == P && N == R && P == J)
      contribution += 1.0*getRDM(I,M,N,K,L,R, rdm );

    else if (M == P && N == R && P == I)
      contribution += -1.0*getRDM(J,M,N,K,L,R, rdm );

    else if (M == P && N == J && L == R)
      contribution += 1.0*getRDM(I,M,N,K,P,R, rdm );

    else if (M == P && N == J && K == R)
      contribution += -1.0*getRDM(I,M,N,L,P,R, rdm );

    else if (M == P && N == J && P == R)
      contribution += -1.0*getRDM(I,M,N,K,L,R, rdm );

    else if (M == P && N == J && P == I)
      contribution += 1.0*getRDM(M,N,R,K,L,R, rdm );

    else if (M == P && N == I && L == R)
      contribution += -1.0*getRDM(J,M,N,K,P,R, rdm );

    else if (M == P && N == I && K == R)
      contribution += 1.0*getRDM(J,M,N,L,P,R, rdm );

    else if (M == P && N == I && P == R)
      contribution += 1.0*getRDM(J,M,N,K,L,R, rdm );

    else if (M == P && N == I && P == J)
      contribution += -1.0*getRDM(M,N,R,K,L,R, rdm );

    else if (M == N && P == J && L == R)
      contribution += 1.0*getRDM(I,M,P,K,N,R, rdm );

    else if (M == N && P == J && K == R)
      contribution += -1.0*getRDM(I,M,P,L,N,R, rdm );

    else if (M == N && P == I && L == R)
      contribution += -1.0*getRDM(J,M,P,K,N,R, rdm );

    else if (M == N && P == I && K == R)
      contribution += 1.0*getRDM(J,M,P,L,N,R, rdm );

    else if (M == N && N == R && P == J)
      contribution += -1.0*getRDM(I,M,P,K,L,R, rdm );

    else if (M == N && N == R && P == I)
      contribution += 1.0*getRDM(J,M,P,K,L,R, rdm );

    else if (M == N && N == J && L == R)
      contribution += -1.0*getRDM(I,M,P,K,P,R, rdm );

    else if (M == N && N == J && K == R)
      contribution += 1.0*getRDM(I,M,P,L,P,R, rdm );

    else if (M == N && N == J && P == R)
      contribution += 1.0*getRDM(I,M,P,K,L,R, rdm );

    else if (M == N && N == J && P == I)
      contribution += -1.0*getRDM(M,P,R,K,L,R, rdm );

    else if (M == N && N == I && L == R)
      contribution += 1.0*getRDM(J,M,P,K,P,R, rdm );

    else if (M == N && N == I && K == R)
      contribution += -1.0*getRDM(J,M,P,L,P,R, rdm );

    else if (M == N && N == I && P == R)
      contribution += -1.0*getRDM(J,M,P,K,L,R, rdm );

    else if (M == N && N == I && P == J)
      contribution += 1.0*getRDM(M,P,R,K,L,R, rdm );

    else if (M == N && N == P && L == R)
      contribution += -1.0*getRDM(I,J,M,K,P,R, rdm );

    else if (M == N && N == P && K == R)
      contribution += 1.0*getRDM(I,J,M,L,P,R, rdm );

    else if (M == N && N == P && P == R)
      contribution += 1.0*getRDM(I,J,M,K,L,R, rdm );

    else if (M == N && N == P && P == J)
      contribution += 1.0*getRDM(I,M,R,K,L,R, rdm );

    else if (M == N && N == P && P == I)
      contribution += -1.0*getRDM(J,M,R,K,L,R, rdm );

    else if (P == J && L == R)
      contribution += -1.0*getRDM(I,M,N,P,K,M,N,R, rdm );

    else if (P == J && K == R)
      contribution += 1.0*getRDM(I,M,N,P,L,M,N,R, rdm );

    else if (P == I && L == R)
      contribution += 1.0*getRDM(J,M,N,P,K,M,N,R, rdm );

    else if (P == I && K == R)
      contribution += -1.0*getRDM(J,M,N,P,L,M,N,R, rdm );

    else if (N == R && P == J)
      contribution += -1.0*getRDM(I,M,N,P,K,L,M,R, rdm );

    else if (N == R && P == I)
      contribution += 1.0*getRDM(J,M,N,P,K,L,M,R, rdm );

    else if (N == J && L == R)
      contribution += 1.0*getRDM(I,M,N,P,K,M,P,R, rdm );

    else if (N == J && K == R)
      contribution += -1.0*getRDM(I,M,N,P,L,M,P,R, rdm );

    else if (N == J && P == R)
      contribution += 1.0*getRDM(I,M,N,P,K,L,M,R, rdm );

    else if (N == J && P == I)
      contribution += 1.0*getRDM(M,N,P,R,K,L,M,R, rdm );

    else if (N == I && L == R)
      contribution += -1.0*getRDM(J,M,N,P,K,M,P,R, rdm );

    else if (N == I && K == R)
      contribution += 1.0*getRDM(J,M,N,P,L,M,P,R, rdm );

    else if (N == I && P == R)
      contribution += -1.0*getRDM(J,M,N,P,K,L,M,R, rdm );

    else if (N == I && P == J)
      contribution += -1.0*getRDM(M,N,P,R,K,L,M,R, rdm );

    else if (N == P && L == R)
      contribution += -1.0*getRDM(I,J,M,N,K,M,P,R, rdm );

    else if (N == P && K == R)
      contribution += 1.0*getRDM(I,J,M,N,L,M,P,R, rdm );

    else if (N == P && P == R)
      contribution += -1.0*getRDM(I,J,M,N,K,L,M,R, rdm );

    else if (N == P && P == J)
      contribution += 1.0*getRDM(I,M,N,R,K,L,M,R, rdm );

    else if (N == P && P == I)
      contribution += -1.0*getRDM(J,M,N,R,K,L,M,R, rdm );

    else if (M == R && P == J)
      contribution += 1.0*getRDM(I,M,N,P,K,L,N,R, rdm );

    else if (M == R && P == I)
      contribution += -1.0*getRDM(J,M,N,P,K,L,N,R, rdm );

    else if (M == R && N == J)
      contribution += -1.0*getRDM(I,M,N,P,K,L,P,R, rdm );

    else if (M == R && N == I)
      contribution += 1.0*getRDM(J,M,N,P,K,L,P,R, rdm );

    else if (M == R && N == P)
      contribution += 1.0*getRDM(I,J,M,N,K,L,P,R, rdm );

    else if (M == J && L == R)
      contribution += -1.0*getRDM(I,M,N,P,K,N,P,R, rdm );

    else if (M == J && K == R)
      contribution += 1.0*getRDM(I,M,N,P,L,N,P,R, rdm );

    else if (M == J && P == R)
      contribution += -1.0*getRDM(I,M,N,P,K,L,N,R, rdm );

    else if (M == J && P == I)
      contribution += -1.0*getRDM(M,N,P,R,K,L,N,R, rdm );

    else if (M == J && N == R)
      contribution += 1.0*getRDM(I,M,N,P,K,L,P,R, rdm );

    else if (M == J && N == I)
      contribution += 1.0*getRDM(M,N,P,R,K,L,P,R, rdm );

    else if (M == J && N == P)
      contribution += -1.0*getRDM(I,M,N,R,K,L,P,R, rdm );

    else if (M == I && L == R)
      contribution += 1.0*getRDM(J,M,N,P,K,N,P,R, rdm );

    else if (M == I && K == R)
      contribution += -1.0*getRDM(J,M,N,P,L,N,P,R, rdm );

    else if (M == I && P == R)
      contribution += 1.0*getRDM(J,M,N,P,K,L,N,R, rdm );

    else if (M == I && P == J)
      contribution += 1.0*getRDM(M,N,P,R,K,L,N,R, rdm );

    else if (M == I && N == R)
      contribution += -1.0*getRDM(J,M,N,P,K,L,P,R, rdm );

    else if (M == I && N == J)
      contribution += -1.0*getRDM(M,N,P,R,K,L,P,R, rdm );

    else if (M == I && N == P)
      contribution += 1.0*getRDM(J,M,N,R,K,L,P,R, rdm );

    else if (M == P && L == R)
      contribution += 1.0*getRDM(I,J,M,N,K,N,P,R, rdm );

    else if (M == P && K == R)
      contribution += -1.0*getRDM(I,J,M,N,L,N,P,R, rdm );

    else if (M == P && P == R)
      contribution += 1.0*getRDM(I,J,M,N,K,L,N,R, rdm );

    else if (M == P && P == J)
      contribution += -1.0*getRDM(I,M,N,R,K,L,N,R, rdm );

    else if (M == P && P == I)
      contribution += 1.0*getRDM(J,M,N,R,K,L,N,R, rdm );

    else if (M == P && N == R)
      contribution += -1.0*getRDM(I,J,M,N,K,L,P,R, rdm );

    else if (M == P && N == J)
      contribution += 1.0*getRDM(I,M,N,R,K,L,P,R, rdm );

    else if (M == P && N == I)
      contribution += -1.0*getRDM(J,M,N,R,K,L,P,R, rdm );

    else if (M == N && L == R)
      contribution += -1.0*getRDM(I,J,M,P,K,N,P,R, rdm );

    else if (M == N && K == R)
      contribution += 1.0*getRDM(I,J,M,P,L,N,P,R, rdm );

    else if (M == N && P == R)
      contribution += -1.0*getRDM(I,J,M,P,K,L,N,R, rdm );

    else if (M == N && P == J)
      contribution += 1.0*getRDM(I,M,P,R,K,L,N,R, rdm );

    else if (M == N && P == I)
      contribution += -1.0*getRDM(J,M,P,R,K,L,N,R, rdm );

    else if (M == N && N == R)
      contribution += 1.0*getRDM(I,J,M,P,K,L,P,R, rdm );

    else if (M == N && N == J)
      contribution += -1.0*getRDM(I,M,P,R,K,L,P,R, rdm );

    else if (M == N && N == I)
      contribution += 1.0*getRDM(J,M,P,R,K,L,P,R, rdm );

    else if (M == N && N == P)
      contribution += -1.0*getRDM(I,J,M,R,K,L,P,R, rdm );

    else if (L == R)
      contribution += 1.0*getRDM(I,J,M,N,P,K,M,N,P,R, rdm );

    else if (K == R)
      contribution += -1.0*getRDM(I,J,M,N,P,L,M,N,P,R, rdm );

    else if (P == R)
      contribution += -1.0*getRDM(I,J,M,N,P,K,L,M,N,R, rdm );

    else if (P == J)
      contribution += -1.0*getRDM(I,M,N,P,R,K,L,M,N,R, rdm );

    else if (P == I)
      contribution += 1.0*getRDM(J,M,N,P,R,K,L,M,N,R, rdm );

    else if (N == R)
      contribution += 1.0*getRDM(I,J,M,N,P,K,L,M,P,R, rdm );

    else if (N == J)
      contribution += 1.0*getRDM(I,M,N,P,R,K,L,M,P,R, rdm );

    else if (N == I)
      contribution += -1.0*getRDM(J,M,N,P,R,K,L,M,P,R, rdm );

    else if (N == P)
      contribution += -1.0*getRDM(I,J,M,N,R,K,L,M,P,R, rdm );

    else if (M == R)
      contribution += -1.0*getRDM(I,J,M,N,P,K,L,N,P,R, rdm );

    else if (M == J)
      contribution += -1.0*getRDM(I,M,N,P,R,K,L,N,P,R, rdm );

    else if (M == I)
      contribution += 1.0*getRDM(J,M,N,P,R,K,L,N,P,R, rdm );

    else if (M == P)
      contribution += 1.0*getRDM(I,J,M,N,R,K,L,N,P,R, rdm );

    else if (M == N)
      contribution += -1.0*getRDM(I,J,M,P,R,K,L,N,P,R, rdm );

    else
      contribution += 1.0*getRDM(I,J,M,N,P,R,K,L,M,N,P,R, rdm );

    return contribution;
    
  }
  
  complexT calcTermGJ2l(int M, int N, int P, int Q, int I, int J, int K, int L, int R, MatrixXcT& rdm) {
    complexT contribution = complexT(0.,0.);
    if (M == Q && N == P && P == J && Q == I && L == R)
      contribution += 1.0*getRDM(M,N,K,R, rdm );

    else if (M == Q && N == P && P == J && Q == I && K == R)
      contribution += -1.0*getRDM(M,N,L,R, rdm );

    else if (M == P && N == Q && P == J && Q == I && L == R)
      contribution += -1.0*getRDM(M,N,K,R, rdm );

    else if (M == P && N == Q && P == J && Q == I && K == R)
      contribution += 1.0*getRDM(M,N,L,R, rdm );

    else if (N == Q && P == J && Q == I && L == R)
      contribution += -1.0*getRDM(M,N,P,K,M,R, rdm );

    else if (N == Q && P == J && Q == I && K == R)
      contribution += 1.0*getRDM(M,N,P,L,M,R, rdm );

    else if (N == P && P == J && Q == I && L == R)
      contribution += 1.0*getRDM(M,N,Q,K,M,R, rdm );

    else if (N == P && P == J && Q == I && K == R)
      contribution += -1.0*getRDM(M,N,Q,L,M,R, rdm );

    else if (M == R && N == Q && P == J && Q == I)
      contribution += 1.0*getRDM(M,N,P,K,L,R, rdm );

    else if (M == R && N == P && P == J && Q == I)
      contribution += -1.0*getRDM(M,N,Q,K,L,R, rdm );

    else if (M == J && N == Q && Q == I && L == R)
      contribution += 1.0*getRDM(M,N,P,K,P,R, rdm );

    else if (M == J && N == Q && Q == I && K == R)
      contribution += -1.0*getRDM(M,N,P,L,P,R, rdm );

    else if (M == J && N == Q && P == R && Q == I)
      contribution += -1.0*getRDM(M,N,P,K,L,R, rdm );

    else if (M == J && N == P && Q == I && L == R)
      contribution += -1.0*getRDM(M,N,Q,K,P,R, rdm );

    else if (M == J && N == P && Q == I && K == R)
      contribution += 1.0*getRDM(M,N,Q,L,P,R, rdm );

    else if (M == J && N == P && P == R && Q == I)
      contribution += 1.0*getRDM(M,N,Q,K,L,R, rdm );

    else if (M == I && N == Q && P == J && L == R)
      contribution += 1.0*getRDM(M,N,P,K,Q,R, rdm );

    else if (M == I && N == Q && P == J && K == R)
      contribution += -1.0*getRDM(M,N,P,L,Q,R, rdm );

    else if (M == I && N == Q && P == J && Q == R)
      contribution += -1.0*getRDM(M,N,P,K,L,R, rdm );

    else if (M == I && N == P && P == J && L == R)
      contribution += -1.0*getRDM(M,N,Q,K,Q,R, rdm );

    else if (M == I && N == P && P == J && K == R)
      contribution += 1.0*getRDM(M,N,Q,L,Q,R, rdm );

    else if (M == I && N == P && P == J && Q == R)
      contribution += 1.0*getRDM(M,N,Q,K,L,R, rdm );

    else if (M == Q && P == J && Q == I && L == R)
      contribution += 1.0*getRDM(M,N,P,K,N,R, rdm );

    else if (M == Q && P == J && Q == I && K == R)
      contribution += -1.0*getRDM(M,N,P,L,N,R, rdm );

    else if (M == Q && N == R && P == J && Q == I)
      contribution += -1.0*getRDM(M,N,P,K,L,R, rdm );

    else if (M == Q && N == J && Q == I && L == R)
      contribution += -1.0*getRDM(M,N,P,K,P,R, rdm );

    else if (M == Q && N == J && Q == I && K == R)
      contribution += 1.0*getRDM(M,N,P,L,P,R, rdm );

    else if (M == Q && N == J && P == R && Q == I)
      contribution += 1.0*getRDM(M,N,P,K,L,R, rdm );

    else if (M == Q && N == I && P == J && L == R)
      contribution += -1.0*getRDM(M,N,P,K,Q,R, rdm );

    else if (M == Q && N == I && P == J && K == R)
      contribution += 1.0*getRDM(M,N,P,L,Q,R, rdm );

    else if (M == Q && N == I && P == J && Q == R)
      contribution += 1.0*getRDM(M,N,P,K,L,R, rdm );

    else if (M == Q && N == P && Q == I && L == R)
      contribution += 1.0*getRDM(J,M,N,K,P,R, rdm );

    else if (M == Q && N == P && Q == I && K == R)
      contribution += -1.0*getRDM(J,M,N,L,P,R, rdm );

    else if (M == Q && N == P && P == R && Q == I)
      contribution += -1.0*getRDM(J,M,N,K,L,R, rdm );

    else if (M == Q && N == P && P == J && L == R)
      contribution += 1.0*getRDM(I,M,N,K,Q,R, rdm );

    else if (M == Q && N == P && P == J && K == R)
      contribution += -1.0*getRDM(I,M,N,L,Q,R, rdm );

    else if (M == Q && N == P && P == J && Q == R)
      contribution += -1.0*getRDM(I,M,N,K,L,R, rdm );

    else if (M == Q && N == P && P == J && Q == I)
      contribution += 1.0*getRDM(M,N,R,K,L,R, rdm );

    else if (M == P && P == J && Q == I && L == R)
      contribution += -1.0*getRDM(M,N,Q,K,N,R, rdm );

    else if (M == P && P == J && Q == I && K == R)
      contribution += 1.0*getRDM(M,N,Q,L,N,R, rdm );

    else if (M == P && N == R && P == J && Q == I)
      contribution += 1.0*getRDM(M,N,Q,K,L,R, rdm );

    else if (M == P && N == J && Q == I && L == R)
      contribution += 1.0*getRDM(M,N,Q,K,P,R, rdm );

    else if (M == P && N == J && Q == I && K == R)
      contribution += -1.0*getRDM(M,N,Q,L,P,R, rdm );

    else if (M == P && N == J && P == R && Q == I)
      contribution += -1.0*getRDM(M,N,Q,K,L,R, rdm );

    else if (M == P && N == I && P == J && L == R)
      contribution += 1.0*getRDM(M,N,Q,K,Q,R, rdm );

    else if (M == P && N == I && P == J && K == R)
      contribution += -1.0*getRDM(M,N,Q,L,Q,R, rdm );

    else if (M == P && N == I && P == J && Q == R)
      contribution += -1.0*getRDM(M,N,Q,K,L,R, rdm );

    else if (M == P && N == Q && Q == I && L == R)
      contribution += -1.0*getRDM(J,M,N,K,P,R, rdm );

    else if (M == P && N == Q && Q == I && K == R)
      contribution += 1.0*getRDM(J,M,N,L,P,R, rdm );

    else if (M == P && N == Q && P == R && Q == I)
      contribution += 1.0*getRDM(J,M,N,K,L,R, rdm );

    else if (M == P && N == Q && P == J && L == R)
      contribution += -1.0*getRDM(I,M,N,K,Q,R, rdm );

    else if (M == P && N == Q && P == J && K == R)
      contribution += 1.0*getRDM(I,M,N,L,Q,R, rdm );

    else if (M == P && N == Q && P == J && Q == R)
      contribution += 1.0*getRDM(I,M,N,K,L,R, rdm );

    else if (M == P && N == Q && P == J && Q == I)
      contribution += -1.0*getRDM(M,N,R,K,L,R, rdm );

    else if (P == J && Q == I && L == R)
      contribution += 1.0*getRDM(M,N,P,Q,K,M,N,R, rdm );

    else if (P == J && Q == I && K == R)
      contribution += -1.0*getRDM(M,N,P,Q,L,M,N,R, rdm );

    else if (N == R && P == J && Q == I)
      contribution += 1.0*getRDM(M,N,P,Q,K,L,M,R, rdm );

    else if (N == J && Q == I && L == R)
      contribution += -1.0*getRDM(M,N,P,Q,K,M,P,R, rdm );

    else if (N == J && Q == I && K == R)
      contribution += 1.0*getRDM(M,N,P,Q,L,M,P,R, rdm );

    else if (N == J && P == R && Q == I)
      contribution += -1.0*getRDM(M,N,P,Q,K,L,M,R, rdm );

    else if (N == I && P == J && L == R)
      contribution += -1.0*getRDM(M,N,P,Q,K,M,Q,R, rdm );

    else if (N == I && P == J && K == R)
      contribution += 1.0*getRDM(M,N,P,Q,L,M,Q,R, rdm );

    else if (N == I && P == J && Q == R)
      contribution += -1.0*getRDM(M,N,P,Q,K,L,M,R, rdm );

    else if (N == Q && Q == I && L == R)
      contribution += -1.0*getRDM(J,M,N,P,K,M,P,R, rdm );

    else if (N == Q && Q == I && K == R)
      contribution += 1.0*getRDM(J,M,N,P,L,M,P,R, rdm );

    else if (N == Q && P == R && Q == I)
      contribution += -1.0*getRDM(J,M,N,P,K,L,M,R, rdm );

    else if (N == Q && P == J && L == R)
      contribution += -1.0*getRDM(I,M,N,P,K,M,Q,R, rdm );

    else if (N == Q && P == J && K == R)
      contribution += 1.0*getRDM(I,M,N,P,L,M,Q,R, rdm );

    else if (N == Q && P == J && Q == R)
      contribution += -1.0*getRDM(I,M,N,P,K,L,M,R, rdm );

    else if (N == Q && P == J && Q == I)
      contribution += -1.0*getRDM(M,N,P,R,K,L,M,R, rdm );

    else if (N == P && Q == I && L == R)
      contribution += 1.0*getRDM(J,M,N,Q,K,M,P,R, rdm );

    else if (N == P && Q == I && K == R)
      contribution += -1.0*getRDM(J,M,N,Q,L,M,P,R, rdm );

    else if (N == P && P == R && Q == I)
      contribution += 1.0*getRDM(J,M,N,Q,K,L,M,R, rdm );

    else if (N == P && P == J && L == R)
      contribution += 1.0*getRDM(I,M,N,Q,K,M,Q,R, rdm );

    else if (N == P && P == J && K == R)
      contribution += -1.0*getRDM(I,M,N,Q,L,M,Q,R, rdm );

    else if (N == P && P == J && Q == R)
      contribution += 1.0*getRDM(I,M,N,Q,K,L,M,R, rdm );

    else if (N == P && P == J && Q == I)
      contribution += 1.0*getRDM(M,N,Q,R,K,L,M,R, rdm );

    else if (M == R && P == J && Q == I)
      contribution += -1.0*getRDM(M,N,P,Q,K,L,N,R, rdm );

    else if (M == R && N == J && Q == I)
      contribution += 1.0*getRDM(M,N,P,Q,K,L,P,R, rdm );

    else if (M == R && N == I && P == J)
      contribution += 1.0*getRDM(M,N,P,Q,K,L,Q,R, rdm );

    else if (M == R && N == Q && Q == I)
      contribution += 1.0*getRDM(J,M,N,P,K,L,P,R, rdm );

    else if (M == R && N == Q && P == J)
      contribution += 1.0*getRDM(I,M,N,P,K,L,Q,R, rdm );

    else if (M == R && N == P && Q == I)
      contribution += -1.0*getRDM(J,M,N,Q,K,L,P,R, rdm );

    else if (M == R && N == P && P == J)
      contribution += -1.0*getRDM(I,M,N,Q,K,L,Q,R, rdm );

    else if (M == J && Q == I && L == R)
      contribution += 1.0*getRDM(M,N,P,Q,K,N,P,R, rdm );

    else if (M == J && Q == I && K == R)
      contribution += -1.0*getRDM(M,N,P,Q,L,N,P,R, rdm );

    else if (M == J && P == R && Q == I)
      contribution += 1.0*getRDM(M,N,P,Q,K,L,N,R, rdm );

    else if (M == J && N == R && Q == I)
      contribution += -1.0*getRDM(M,N,P,Q,K,L,P,R, rdm );

    else if (M == J && N == I && L == R)
      contribution += 1.0*getRDM(M,N,P,Q,K,P,Q,R, rdm );

    else if (M == J && N == I && K == R)
      contribution += -1.0*getRDM(M,N,P,Q,L,P,Q,R, rdm );

    else if (M == J && N == I && Q == R)
      contribution += 1.0*getRDM(M,N,P,Q,K,L,P,R, rdm );

    else if (M == J && N == I && P == R)
      contribution += -1.0*getRDM(M,N,P,Q,K,L,Q,R, rdm );

    else if (M == J && N == Q && L == R)
      contribution += 1.0*getRDM(I,M,N,P,K,P,Q,R, rdm );

    else if (M == J && N == Q && K == R)
      contribution += -1.0*getRDM(I,M,N,P,L,P,Q,R, rdm );

    else if (M == J && N == Q && Q == R)
      contribution += 1.0*getRDM(I,M,N,P,K,L,P,R, rdm );

    else if (M == J && N == Q && Q == I)
      contribution += 1.0*getRDM(M,N,P,R,K,L,P,R, rdm );

    else if (M == J && N == Q && P == R)
      contribution += -1.0*getRDM(I,M,N,P,K,L,Q,R, rdm );

    else if (M == J && N == P && L == R)
      contribution += -1.0*getRDM(I,M,N,Q,K,P,Q,R, rdm );

    else if (M == J && N == P && K == R)
      contribution += 1.0*getRDM(I,M,N,Q,L,P,Q,R, rdm );

    else if (M == J && N == P && Q == R)
      contribution += -1.0*getRDM(I,M,N,Q,K,L,P,R, rdm );

    else if (M == J && N == P && Q == I)
      contribution += -1.0*getRDM(M,N,Q,R,K,L,P,R, rdm );

    else if (M == J && N == P && P == R)
      contribution += 1.0*getRDM(I,M,N,Q,K,L,Q,R, rdm );

    else if (M == I && P == J && L == R)
      contribution += 1.0*getRDM(M,N,P,Q,K,N,Q,R, rdm );

    else if (M == I && P == J && K == R)
      contribution += -1.0*getRDM(M,N,P,Q,L,N,Q,R, rdm );

    else if (M == I && P == J && Q == R)
      contribution += 1.0*getRDM(M,N,P,Q,K,L,N,R, rdm );

    else if (M == I && N == R && P == J)
      contribution += -1.0*getRDM(M,N,P,Q,K,L,Q,R, rdm );

    else if (M == I && N == J && L == R)
      contribution += -1.0*getRDM(M,N,P,Q,K,P,Q,R, rdm );

    else if (M == I && N == J && K == R)
      contribution += 1.0*getRDM(M,N,P,Q,L,P,Q,R, rdm );

    else if (M == I && N == J && Q == R)
      contribution += -1.0*getRDM(M,N,P,Q,K,L,P,R, rdm );

    else if (M == I && N == J && P == R)
      contribution += 1.0*getRDM(M,N,P,Q,K,L,Q,R, rdm );

    else if (M == I && N == Q && L == R)
      contribution += -1.0*getRDM(J,M,N,P,K,P,Q,R, rdm );

    else if (M == I && N == Q && K == R)
      contribution += 1.0*getRDM(J,M,N,P,L,P,Q,R, rdm );

    else if (M == I && N == Q && Q == R)
      contribution += -1.0*getRDM(J,M,N,P,K,L,P,R, rdm );

    else if (M == I && N == Q && P == R)
      contribution += 1.0*getRDM(J,M,N,P,K,L,Q,R, rdm );

    else if (M == I && N == Q && P == J)
      contribution += 1.0*getRDM(M,N,P,R,K,L,Q,R, rdm );

    else if (M == I && N == P && L == R)
      contribution += 1.0*getRDM(J,M,N,Q,K,P,Q,R, rdm );

    else if (M == I && N == P && K == R)
      contribution += -1.0*getRDM(J,M,N,Q,L,P,Q,R, rdm );

    else if (M == I && N == P && Q == R)
      contribution += 1.0*getRDM(J,M,N,Q,K,L,P,R, rdm );

    else if (M == I && N == P && P == R)
      contribution += -1.0*getRDM(J,M,N,Q,K,L,Q,R, rdm );

    else if (M == I && N == P && P == J)
      contribution += -1.0*getRDM(M,N,Q,R,K,L,Q,R, rdm );

    else if (M == Q && Q == I && L == R)
      contribution += 1.0*getRDM(J,M,N,P,K,N,P,R, rdm );

    else if (M == Q && Q == I && K == R)
      contribution += -1.0*getRDM(J,M,N,P,L,N,P,R, rdm );

    else if (M == Q && P == R && Q == I)
      contribution += 1.0*getRDM(J,M,N,P,K,L,N,R, rdm );

    else if (M == Q && P == J && L == R)
      contribution += 1.0*getRDM(I,M,N,P,K,N,Q,R, rdm );

    else if (M == Q && P == J && K == R)
      contribution += -1.0*getRDM(I,M,N,P,L,N,Q,R, rdm );

    else if (M == Q && P == J && Q == R)
      contribution += 1.0*getRDM(I,M,N,P,K,L,N,R, rdm );

    else if (M == Q && P == J && Q == I)
      contribution += 1.0*getRDM(M,N,P,R,K,L,N,R, rdm );

    else if (M == Q && N == R && Q == I)
      contribution += -1.0*getRDM(J,M,N,P,K,L,P,R, rdm );

    else if (M == Q && N == R && P == J)
      contribution += -1.0*getRDM(I,M,N,P,K,L,Q,R, rdm );

    else if (M == Q && N == J && L == R)
      contribution += -1.0*getRDM(I,M,N,P,K,P,Q,R, rdm );

    else if (M == Q && N == J && K == R)
      contribution += 1.0*getRDM(I,M,N,P,L,P,Q,R, rdm );

    else if (M == Q && N == J && Q == R)
      contribution += -1.0*getRDM(I,M,N,P,K,L,P,R, rdm );

    else if (M == Q && N == J && Q == I)
      contribution += -1.0*getRDM(M,N,P,R,K,L,P,R, rdm );

    else if (M == Q && N == J && P == R)
      contribution += 1.0*getRDM(I,M,N,P,K,L,Q,R, rdm );

    else if (M == Q && N == I && L == R)
      contribution += 1.0*getRDM(J,M,N,P,K,P,Q,R, rdm );

    else if (M == Q && N == I && K == R)
      contribution += -1.0*getRDM(J,M,N,P,L,P,Q,R, rdm );

    else if (M == Q && N == I && Q == R)
      contribution += 1.0*getRDM(J,M,N,P,K,L,P,R, rdm );

    else if (M == Q && N == I && P == R)
      contribution += -1.0*getRDM(J,M,N,P,K,L,Q,R, rdm );

    else if (M == Q && N == I && P == J)
      contribution += -1.0*getRDM(M,N,P,R,K,L,Q,R, rdm );

    else if (M == Q && N == P && L == R)
      contribution += 1.0*getRDM(I,J,M,N,K,P,Q,R, rdm );

    else if (M == Q && N == P && K == R)
      contribution += -1.0*getRDM(I,J,M,N,L,P,Q,R, rdm );

    else if (M == Q && N == P && Q == R)
      contribution += 1.0*getRDM(I,J,M,N,K,L,P,R, rdm );

    else if (M == Q && N == P && Q == I)
      contribution += 1.0*getRDM(J,M,N,R,K,L,P,R, rdm );

    else if (M == Q && N == P && P == R)
      contribution += -1.0*getRDM(I,J,M,N,K,L,Q,R, rdm );

    else if (M == Q && N == P && P == J)
      contribution += 1.0*getRDM(I,M,N,R,K,L,Q,R, rdm );

    else if (M == P && Q == I && L == R)
      contribution += -1.0*getRDM(J,M,N,Q,K,N,P,R, rdm );

    else if (M == P && Q == I && K == R)
      contribution += 1.0*getRDM(J,M,N,Q,L,N,P,R, rdm );

    else if (M == P && P == R && Q == I)
      contribution += -1.0*getRDM(J,M,N,Q,K,L,N,R, rdm );

    else if (M == P && P == J && L == R)
      contribution += -1.0*getRDM(I,M,N,Q,K,N,Q,R, rdm );

    else if (M == P && P == J && K == R)
      contribution += 1.0*getRDM(I,M,N,Q,L,N,Q,R, rdm );

    else if (M == P && P == J && Q == R)
      contribution += -1.0*getRDM(I,M,N,Q,K,L,N,R, rdm );

    else if (M == P && P == J && Q == I)
      contribution += -1.0*getRDM(M,N,Q,R,K,L,N,R, rdm );

    else if (M == P && N == R && Q == I)
      contribution += 1.0*getRDM(J,M,N,Q,K,L,P,R, rdm );

    else if (M == P && N == R && P == J)
      contribution += 1.0*getRDM(I,M,N,Q,K,L,Q,R, rdm );

    else if (M == P && N == J && L == R)
      contribution += 1.0*getRDM(I,M,N,Q,K,P,Q,R, rdm );

    else if (M == P && N == J && K == R)
      contribution += -1.0*getRDM(I,M,N,Q,L,P,Q,R, rdm );

    else if (M == P && N == J && Q == R)
      contribution += 1.0*getRDM(I,M,N,Q,K,L,P,R, rdm );

    else if (M == P && N == J && Q == I)
      contribution += 1.0*getRDM(M,N,Q,R,K,L,P,R, rdm );

    else if (M == P && N == J && P == R)
      contribution += -1.0*getRDM(I,M,N,Q,K,L,Q,R, rdm );

    else if (M == P && N == I && L == R)
      contribution += -1.0*getRDM(J,M,N,Q,K,P,Q,R, rdm );

    else if (M == P && N == I && K == R)
      contribution += 1.0*getRDM(J,M,N,Q,L,P,Q,R, rdm );

    else if (M == P && N == I && Q == R)
      contribution += -1.0*getRDM(J,M,N,Q,K,L,P,R, rdm );

    else if (M == P && N == I && P == R)
      contribution += 1.0*getRDM(J,M,N,Q,K,L,Q,R, rdm );

    else if (M == P && N == I && P == J)
      contribution += 1.0*getRDM(M,N,Q,R,K,L,Q,R, rdm );

    else if (M == P && N == Q && L == R)
      contribution += -1.0*getRDM(I,J,M,N,K,P,Q,R, rdm );

    else if (M == P && N == Q && K == R)
      contribution += 1.0*getRDM(I,J,M,N,L,P,Q,R, rdm );

    else if (M == P && N == Q && Q == R)
      contribution += -1.0*getRDM(I,J,M,N,K,L,P,R, rdm );

    else if (M == P && N == Q && Q == I)
      contribution += -1.0*getRDM(J,M,N,R,K,L,P,R, rdm );

    else if (M == P && N == Q && P == R)
      contribution += 1.0*getRDM(I,J,M,N,K,L,Q,R, rdm );

    else if (M == P && N == Q && P == J)
      contribution += -1.0*getRDM(I,M,N,R,K,L,Q,R, rdm );

    else if (Q == I && L == R)
      contribution += 1.0*getRDM(J,M,N,P,Q,K,M,N,P,R, rdm );

    else if (Q == I && K == R)
      contribution += -1.0*getRDM(J,M,N,P,Q,L,M,N,P,R, rdm );

    else if (P == R && Q == I)
      contribution += -1.0*getRDM(J,M,N,P,Q,K,L,M,N,R, rdm );

    else if (P == J && L == R)
      contribution += 1.0*getRDM(I,M,N,P,Q,K,M,N,Q,R, rdm );

    else if (P == J && K == R)
      contribution += -1.0*getRDM(I,M,N,P,Q,L,M,N,Q,R, rdm );

    else if (P == J && Q == R)
      contribution += -1.0*getRDM(I,M,N,P,Q,K,L,M,N,R, rdm );

    else if (P == J && Q == I)
      contribution += 1.0*getRDM(M,N,P,Q,R,K,L,M,N,R, rdm );

    else if (N == R && Q == I)
      contribution += 1.0*getRDM(J,M,N,P,Q,K,L,M,P,R, rdm );

    else if (N == R && P == J)
      contribution += 1.0*getRDM(I,M,N,P,Q,K,L,M,Q,R, rdm );

    else if (N == J && L == R)
      contribution += -1.0*getRDM(I,M,N,P,Q,K,M,P,Q,R, rdm );

    else if (N == J && K == R)
      contribution += 1.0*getRDM(I,M,N,P,Q,L,M,P,Q,R, rdm );

    else if (N == J && Q == R)
      contribution += 1.0*getRDM(I,M,N,P,Q,K,L,M,P,R, rdm );

    else if (N == J && Q == I)
      contribution += -1.0*getRDM(M,N,P,Q,R,K,L,M,P,R, rdm );

    else if (N == J && P == R)
      contribution += -1.0*getRDM(I,M,N,P,Q,K,L,M,Q,R, rdm );

    else if (N == I && L == R)
      contribution += 1.0*getRDM(J,M,N,P,Q,K,M,P,Q,R, rdm );

    else if (N == I && K == R)
      contribution += -1.0*getRDM(J,M,N,P,Q,L,M,P,Q,R, rdm );

    else if (N == I && Q == R)
      contribution += -1.0*getRDM(J,M,N,P,Q,K,L,M,P,R, rdm );

    else if (N == I && P == R)
      contribution += 1.0*getRDM(J,M,N,P,Q,K,L,M,Q,R, rdm );

    else if (N == I && P == J)
      contribution += -1.0*getRDM(M,N,P,Q,R,K,L,M,Q,R, rdm );

    else if (N == Q && L == R)
      contribution += -1.0*getRDM(I,J,M,N,P,K,M,P,Q,R, rdm );

    else if (N == Q && K == R)
      contribution += 1.0*getRDM(I,J,M,N,P,L,M,P,Q,R, rdm );

    else if (N == Q && Q == R)
      contribution += 1.0*getRDM(I,J,M,N,P,K,L,M,P,R, rdm );

    else if (N == Q && Q == I)
      contribution += -1.0*getRDM(J,M,N,P,R,K,L,M,P,R, rdm );

    else if (N == Q && P == R)
      contribution += -1.0*getRDM(I,J,M,N,P,K,L,M,Q,R, rdm );

    else if (N == Q && P == J)
      contribution += -1.0*getRDM(I,M,N,P,R,K,L,M,Q,R, rdm );

    else if (N == P && L == R)
      contribution += 1.0*getRDM(I,J,M,N,Q,K,M,P,Q,R, rdm );

    else if (N == P && K == R)
      contribution += -1.0*getRDM(I,J,M,N,Q,L,M,P,Q,R, rdm );

    else if (N == P && Q == R)
      contribution += -1.0*getRDM(I,J,M,N,Q,K,L,M,P,R, rdm );

    else if (N == P && Q == I)
      contribution += 1.0*getRDM(J,M,N,Q,R,K,L,M,P,R, rdm );

    else if (N == P && P == R)
      contribution += 1.0*getRDM(I,J,M,N,Q,K,L,M,Q,R, rdm );

    else if (N == P && P == J)
      contribution += 1.0*getRDM(I,M,N,Q,R,K,L,M,Q,R, rdm );

    else if (M == R && Q == I)
      contribution += -1.0*getRDM(J,M,N,P,Q,K,L,N,P,R, rdm );

    else if (M == R && P == J)
      contribution += -1.0*getRDM(I,M,N,P,Q,K,L,N,Q,R, rdm );

    else if (M == R && N == J)
      contribution += 1.0*getRDM(I,M,N,P,Q,K,L,P,Q,R, rdm );

    else if (M == R && N == I)
      contribution += -1.0*getRDM(J,M,N,P,Q,K,L,P,Q,R, rdm );

    else if (M == R && N == Q)
      contribution += 1.0*getRDM(I,J,M,N,P,K,L,P,Q,R, rdm );

    else if (M == R && N == P)
      contribution += -1.0*getRDM(I,J,M,N,Q,K,L,P,Q,R, rdm );

    else if (M == J && L == R)
      contribution += 1.0*getRDM(I,M,N,P,Q,K,N,P,Q,R, rdm );

    else if (M == J && K == R)
      contribution += -1.0*getRDM(I,M,N,P,Q,L,N,P,Q,R, rdm );

    else if (M == J && Q == R)
      contribution += -1.0*getRDM(I,M,N,P,Q,K,L,N,P,R, rdm );

    else if (M == J && Q == I)
      contribution += 1.0*getRDM(M,N,P,Q,R,K,L,N,P,R, rdm );

    else if (M == J && P == R)
      contribution += 1.0*getRDM(I,M,N,P,Q,K,L,N,Q,R, rdm );

    else if (M == J && N == R)
      contribution += -1.0*getRDM(I,M,N,P,Q,K,L,P,Q,R, rdm );

    else if (M == J && N == I)
      contribution += 1.0*getRDM(M,N,P,Q,R,K,L,P,Q,R, rdm );

    else if (M == J && N == Q)
      contribution += 1.0*getRDM(I,M,N,P,R,K,L,P,Q,R, rdm );

    else if (M == J && N == P)
      contribution += -1.0*getRDM(I,M,N,Q,R,K,L,P,Q,R, rdm );

    else if (M == I && L == R)
      contribution += -1.0*getRDM(J,M,N,P,Q,K,N,P,Q,R, rdm );

    else if (M == I && K == R)
      contribution += 1.0*getRDM(J,M,N,P,Q,L,N,P,Q,R, rdm );

    else if (M == I && Q == R)
      contribution += 1.0*getRDM(J,M,N,P,Q,K,L,N,P,R, rdm );

    else if (M == I && P == R)
      contribution += -1.0*getRDM(J,M,N,P,Q,K,L,N,Q,R, rdm );

    else if (M == I && P == J)
      contribution += 1.0*getRDM(M,N,P,Q,R,K,L,N,Q,R, rdm );

    else if (M == I && N == R)
      contribution += 1.0*getRDM(J,M,N,P,Q,K,L,P,Q,R, rdm );

    else if (M == I && N == J)
      contribution += -1.0*getRDM(M,N,P,Q,R,K,L,P,Q,R, rdm );

    else if (M == I && N == Q)
      contribution += -1.0*getRDM(J,M,N,P,R,K,L,P,Q,R, rdm );

    else if (M == I && N == P)
      contribution += 1.0*getRDM(J,M,N,Q,R,K,L,P,Q,R, rdm );

    else if (M == Q && L == R)
      contribution += 1.0*getRDM(I,J,M,N,P,K,N,P,Q,R, rdm );

    else if (M == Q && K == R)
      contribution += -1.0*getRDM(I,J,M,N,P,L,N,P,Q,R, rdm );

    else if (M == Q && Q == R)
      contribution += -1.0*getRDM(I,J,M,N,P,K,L,N,P,R, rdm );

    else if (M == Q && Q == I)
      contribution += 1.0*getRDM(J,M,N,P,R,K,L,N,P,R, rdm );

    else if (M == Q && P == R)
      contribution += 1.0*getRDM(I,J,M,N,P,K,L,N,Q,R, rdm );

    else if (M == Q && P == J)
      contribution += 1.0*getRDM(I,M,N,P,R,K,L,N,Q,R, rdm );

    else if (M == Q && N == R)
      contribution += -1.0*getRDM(I,J,M,N,P,K,L,P,Q,R, rdm );

    else if (M == Q && N == J)
      contribution += -1.0*getRDM(I,M,N,P,R,K,L,P,Q,R, rdm );

    else if (M == Q && N == I)
      contribution += 1.0*getRDM(J,M,N,P,R,K,L,P,Q,R, rdm );

    else if (M == Q && N == P)
      contribution += 1.0*getRDM(I,J,M,N,R,K,L,P,Q,R, rdm );

    else if (M == P && L == R)
      contribution += -1.0*getRDM(I,J,M,N,Q,K,N,P,Q,R, rdm );

    else if (M == P && K == R)
      contribution += 1.0*getRDM(I,J,M,N,Q,L,N,P,Q,R, rdm );

    else if (M == P && Q == R)
      contribution += 1.0*getRDM(I,J,M,N,Q,K,L,N,P,R, rdm );

    else if (M == P && Q == I)
      contribution += -1.0*getRDM(J,M,N,Q,R,K,L,N,P,R, rdm );

    else if (M == P && P == R)
      contribution += -1.0*getRDM(I,J,M,N,Q,K,L,N,Q,R, rdm );

    else if (M == P && P == J)
      contribution += -1.0*getRDM(I,M,N,Q,R,K,L,N,Q,R, rdm );

    else if (M == P && N == R)
      contribution += 1.0*getRDM(I,J,M,N,Q,K,L,P,Q,R, rdm );

    else if (M == P && N == J)
      contribution += 1.0*getRDM(I,M,N,Q,R,K,L,P,Q,R, rdm );

    else if (M == P && N == I)
      contribution += -1.0*getRDM(J,M,N,Q,R,K,L,P,Q,R, rdm );

    else if (M == P && N == Q)
      contribution += -1.0*getRDM(I,J,M,N,R,K,L,P,Q,R, rdm );

    else if (L == R)
      contribution += 1.0*getRDM(I,J,M,N,P,Q,K,M,N,P,Q,R, rdm );

    else if (K == R)
      contribution += -1.0*getRDM(I,J,M,N,P,Q,L,M,N,P,Q,R, rdm );

    else if (Q == R)
      contribution += 1.0*getRDM(I,J,M,N,P,Q,K,L,M,N,P,R, rdm );

    else if (Q == I)
      contribution += 1.0*getRDM(J,M,N,P,Q,R,K,L,M,N,P,R, rdm );

    else if (P == R)
      contribution += -1.0*getRDM(I,J,M,N,P,Q,K,L,M,N,Q,R, rdm );

    else if (P == J)
      contribution += 1.0*getRDM(I,M,N,P,Q,R,K,L,M,N,Q,R, rdm );

    else if (N == R)
      contribution += 1.0*getRDM(I,J,M,N,P,Q,K,L,M,P,Q,R, rdm );

    else if (N == J)
      contribution += -1.0*getRDM(I,M,N,P,Q,R,K,L,M,P,Q,R, rdm );

    else if (N == I)
      contribution += 1.0*getRDM(J,M,N,P,Q,R,K,L,M,P,Q,R, rdm );

    else if (N == Q)
      contribution += -1.0*getRDM(I,J,M,N,P,R,K,L,M,P,Q,R, rdm );

    else if (N == P)
      contribution += 1.0*getRDM(I,J,M,N,Q,R,K,L,M,P,Q,R, rdm );

    else if (M == R)
      contribution += -1.0*getRDM(I,J,M,N,P,Q,K,L,N,P,Q,R, rdm );

    else if (M == J)
      contribution += 1.0*getRDM(I,M,N,P,Q,R,K,L,N,P,Q,R, rdm );

    else if (M == I)
      contribution += -1.0*getRDM(J,M,N,P,Q,R,K,L,N,P,Q,R, rdm );

    else if (M == Q)
      contribution += 1.0*getRDM(I,J,M,N,P,R,K,L,N,P,Q,R, rdm );

    else if (M == P)
      contribution += -1.0*getRDM(I,J,M,N,Q,R,K,L,N,P,Q,R, rdm );

    else
      contribution += 1.0*getRDM(I,J,M,N,P,Q,R,K,L,M,N,P,Q,R, rdm );

    return contribution;
    
  }
  
  complexT calcTermGJ2m(int M, int N, int P, int I, int J, int K, int L, int R, int S, MatrixXcT& rdm) {
    complexT contribution = complexT(0.,0.);
    if (M == J && N == P && P == I && K == S && L == R)
      contribution += 1.0*getRDM(M,N,R,S, rdm );

    else if (M == I && N == P && P == J && K == S && L == R)
      contribution += -1.0*getRDM(M,N,R,S, rdm );

    else if (M == P && N == J && P == I && K == S && L == R)
      contribution += -1.0*getRDM(M,N,R,S, rdm );

    else if (M == P && N == I && P == J && K == S && L == R)
      contribution += 1.0*getRDM(M,N,R,S, rdm );

    else if (N == J && P == I && K == S && L == R)
      contribution += 1.0*getRDM(M,N,P,M,R,S, rdm );

    else if (N == I && P == J && K == S && L == R)
      contribution += -1.0*getRDM(M,N,P,M,R,S, rdm );

    else if (N == P && P == J && K == S && L == R)
      contribution += 1.0*getRDM(I,M,N,M,R,S, rdm );

    else if (N == P && P == I && K == S && L == R)
      contribution += -1.0*getRDM(J,M,N,M,R,S, rdm );

    else if (M == S && N == J && P == I && L == R)
      contribution += -1.0*getRDM(M,N,P,K,R,S, rdm );

    else if (M == S && N == I && P == J && L == R)
      contribution += 1.0*getRDM(M,N,P,K,R,S, rdm );

    else if (M == S && N == P && P == J && L == R)
      contribution += -1.0*getRDM(I,M,N,K,R,S, rdm );

    else if (M == S && N == P && P == I && L == R)
      contribution += 1.0*getRDM(J,M,N,K,R,S, rdm );

    else if (M == R && N == J && P == I && K == S)
      contribution += -1.0*getRDM(M,N,P,L,R,S, rdm );

    else if (M == R && N == I && P == J && K == S)
      contribution += 1.0*getRDM(M,N,P,L,R,S, rdm );

    else if (M == R && N == P && P == J && K == S)
      contribution += -1.0*getRDM(I,M,N,L,R,S, rdm );

    else if (M == R && N == P && P == I && K == S)
      contribution += 1.0*getRDM(J,M,N,L,R,S, rdm );

    else if (M == J && P == I && K == S && L == R)
      contribution += -1.0*getRDM(M,N,P,N,R,S, rdm );

    else if (M == J && N == S && P == I && L == R)
      contribution += 1.0*getRDM(M,N,P,K,R,S, rdm );

    else if (M == J && N == R && P == I && K == S)
      contribution += 1.0*getRDM(M,N,P,L,R,S, rdm );

    else if (M == J && N == I && K == S && L == R)
      contribution += 1.0*getRDM(M,N,P,P,R,S, rdm );

    else if (M == J && N == I && P == S && L == R)
      contribution += -1.0*getRDM(M,N,P,K,R,S, rdm );

    else if (M == J && N == I && P == R && K == S)
      contribution += -1.0*getRDM(M,N,P,L,R,S, rdm );

    else if (M == J && N == P && K == S && L == R)
      contribution += -1.0*getRDM(I,M,N,P,R,S, rdm );

    else if (M == J && N == P && P == S && L == R)
      contribution += 1.0*getRDM(I,M,N,K,R,S, rdm );

    else if (M == J && N == P && P == R && K == S)
      contribution += 1.0*getRDM(I,M,N,L,R,S, rdm );

    else if (M == J && N == P && P == I && L == R)
      contribution += -1.0*getRDM(M,N,S,K,R,S, rdm );

    else if (M == J && N == P && P == I && K == S)
      contribution += -1.0*getRDM(M,N,R,L,R,S, rdm );

    else if (M == I && P == J && K == S && L == R)
      contribution += 1.0*getRDM(M,N,P,N,R,S, rdm );

    else if (M == I && N == S && P == J && L == R)
      contribution += -1.0*getRDM(M,N,P,K,R,S, rdm );

    else if (M == I && N == R && P == J && K == S)
      contribution += -1.0*getRDM(M,N,P,L,R,S, rdm );

    else if (M == I && N == J && K == S && L == R)
      contribution += -1.0*getRDM(M,N,P,P,R,S, rdm );

    else if (M == I && N == J && P == S && L == R)
      contribution += 1.0*getRDM(M,N,P,K,R,S, rdm );

    else if (M == I && N == J && P == R && K == S)
      contribution += 1.0*getRDM(M,N,P,L,R,S, rdm );

    else if (M == I && N == P && K == S && L == R)
      contribution += 1.0*getRDM(J,M,N,P,R,S, rdm );

    else if (M == I && N == P && P == S && L == R)
      contribution += -1.0*getRDM(J,M,N,K,R,S, rdm );

    else if (M == I && N == P && P == R && K == S)
      contribution += -1.0*getRDM(J,M,N,L,R,S, rdm );

    else if (M == I && N == P && P == J && L == R)
      contribution += 1.0*getRDM(M,N,S,K,R,S, rdm );

    else if (M == I && N == P && P == J && K == S)
      contribution += 1.0*getRDM(M,N,R,L,R,S, rdm );

    else if (M == P && P == J && K == S && L == R)
      contribution += -1.0*getRDM(I,M,N,N,R,S, rdm );

    else if (M == P && P == I && K == S && L == R)
      contribution += 1.0*getRDM(J,M,N,N,R,S, rdm );

    else if (M == P && N == S && P == J && L == R)
      contribution += 1.0*getRDM(I,M,N,K,R,S, rdm );

    else if (M == P && N == S && P == I && L == R)
      contribution += -1.0*getRDM(J,M,N,K,R,S, rdm );

    else if (M == P && N == R && P == J && K == S)
      contribution += 1.0*getRDM(I,M,N,L,R,S, rdm );

    else if (M == P && N == R && P == I && K == S)
      contribution += -1.0*getRDM(J,M,N,L,R,S, rdm );

    else if (M == P && N == J && K == S && L == R)
      contribution += 1.0*getRDM(I,M,N,P,R,S, rdm );

    else if (M == P && N == J && P == S && L == R)
      contribution += -1.0*getRDM(I,M,N,K,R,S, rdm );

    else if (M == P && N == J && P == R && K == S)
      contribution += -1.0*getRDM(I,M,N,L,R,S, rdm );

    else if (M == P && N == J && P == I && L == R)
      contribution += 1.0*getRDM(M,N,S,K,R,S, rdm );

    else if (M == P && N == J && P == I && K == S)
      contribution += 1.0*getRDM(M,N,R,L,R,S, rdm );

    else if (M == P && N == I && K == S && L == R)
      contribution += -1.0*getRDM(J,M,N,P,R,S, rdm );

    else if (M == P && N == I && P == S && L == R)
      contribution += 1.0*getRDM(J,M,N,K,R,S, rdm );

    else if (M == P && N == I && P == R && K == S)
      contribution += 1.0*getRDM(J,M,N,L,R,S, rdm );

    else if (M == P && N == I && P == J && L == R)
      contribution += -1.0*getRDM(M,N,S,K,R,S, rdm );

    else if (M == P && N == I && P == J && K == S)
      contribution += -1.0*getRDM(M,N,R,L,R,S, rdm );

    else if (P == J && K == S && L == R)
      contribution += 1.0*getRDM(I,M,N,P,M,N,R,S, rdm );

    else if (P == I && K == S && L == R)
      contribution += -1.0*getRDM(J,M,N,P,M,N,R,S, rdm );

    else if (N == S && P == J && L == R)
      contribution += 1.0*getRDM(I,M,N,P,K,M,R,S, rdm );

    else if (N == S && P == I && L == R)
      contribution += -1.0*getRDM(J,M,N,P,K,M,R,S, rdm );

    else if (N == R && P == J && K == S)
      contribution += 1.0*getRDM(I,M,N,P,L,M,R,S, rdm );

    else if (N == R && P == I && K == S)
      contribution += -1.0*getRDM(J,M,N,P,L,M,R,S, rdm );

    else if (N == J && K == S && L == R)
      contribution += -1.0*getRDM(I,M,N,P,M,P,R,S, rdm );

    else if (N == J && P == S && L == R)
      contribution += -1.0*getRDM(I,M,N,P,K,M,R,S, rdm );

    else if (N == J && P == R && K == S)
      contribution += -1.0*getRDM(I,M,N,P,L,M,R,S, rdm );

    else if (N == J && P == I && L == R)
      contribution += -1.0*getRDM(M,N,P,S,K,M,R,S, rdm );

    else if (N == J && P == I && K == S)
      contribution += -1.0*getRDM(M,N,P,R,L,M,R,S, rdm );

    else if (N == I && K == S && L == R)
      contribution += 1.0*getRDM(J,M,N,P,M,P,R,S, rdm );

    else if (N == I && P == S && L == R)
      contribution += 1.0*getRDM(J,M,N,P,K,M,R,S, rdm );

    else if (N == I && P == R && K == S)
      contribution += 1.0*getRDM(J,M,N,P,L,M,R,S, rdm );

    else if (N == I && P == J && L == R)
      contribution += 1.0*getRDM(M,N,P,S,K,M,R,S, rdm );

    else if (N == I && P == J && K == S)
      contribution += 1.0*getRDM(M,N,P,R,L,M,R,S, rdm );

    else if (N == P && K == S && L == R)
      contribution += 1.0*getRDM(I,J,M,N,M,P,R,S, rdm );

    else if (N == P && P == S && L == R)
      contribution += 1.0*getRDM(I,J,M,N,K,M,R,S, rdm );

    else if (N == P && P == R && K == S)
      contribution += 1.0*getRDM(I,J,M,N,L,M,R,S, rdm );

    else if (N == P && P == J && L == R)
      contribution += -1.0*getRDM(I,M,N,S,K,M,R,S, rdm );

    else if (N == P && P == J && K == S)
      contribution += -1.0*getRDM(I,M,N,R,L,M,R,S, rdm );

    else if (N == P && P == I && L == R)
      contribution += 1.0*getRDM(J,M,N,S,K,M,R,S, rdm );

    else if (N == P && P == I && K == S)
      contribution += 1.0*getRDM(J,M,N,R,L,M,R,S, rdm );

    else if (M == S && P == J && L == R)
      contribution += -1.0*getRDM(I,M,N,P,K,N,R,S, rdm );

    else if (M == S && P == I && L == R)
      contribution += 1.0*getRDM(J,M,N,P,K,N,R,S, rdm );

    else if (M == S && N == R && P == J)
      contribution += 1.0*getRDM(I,M,N,P,K,L,R,S, rdm );

    else if (M == S && N == R && P == I)
      contribution += -1.0*getRDM(J,M,N,P,K,L,R,S, rdm );

    else if (M == S && N == J && L == R)
      contribution += 1.0*getRDM(I,M,N,P,K,P,R,S, rdm );

    else if (M == S && N == J && P == R)
      contribution += -1.0*getRDM(I,M,N,P,K,L,R,S, rdm );

    else if (M == S && N == J && P == I)
      contribution += -1.0*getRDM(M,N,P,R,K,L,R,S, rdm );

    else if (M == S && N == I && L == R)
      contribution += -1.0*getRDM(J,M,N,P,K,P,R,S, rdm );

    else if (M == S && N == I && P == R)
      contribution += 1.0*getRDM(J,M,N,P,K,L,R,S, rdm );

    else if (M == S && N == I && P == J)
      contribution += 1.0*getRDM(M,N,P,R,K,L,R,S, rdm );

    else if (M == S && N == P && L == R)
      contribution += -1.0*getRDM(I,J,M,N,K,P,R,S, rdm );

    else if (M == S && N == P && P == R)
      contribution += 1.0*getRDM(I,J,M,N,K,L,R,S, rdm );

    else if (M == S && N == P && P == J)
      contribution += -1.0*getRDM(I,M,N,R,K,L,R,S, rdm );

    else if (M == S && N == P && P == I)
      contribution += 1.0*getRDM(J,M,N,R,K,L,R,S, rdm );

    else if (M == R && P == J && K == S)
      contribution += -1.0*getRDM(I,M,N,P,L,N,R,S, rdm );

    else if (M == R && P == I && K == S)
      contribution += 1.0*getRDM(J,M,N,P,L,N,R,S, rdm );

    else if (M == R && N == S && P == J)
      contribution += -1.0*getRDM(I,M,N,P,K,L,R,S, rdm );

    else if (M == R && N == S && P == I)
      contribution += 1.0*getRDM(J,M,N,P,K,L,R,S, rdm );

    else if (M == R && N == J && K == S)
      contribution += 1.0*getRDM(I,M,N,P,L,P,R,S, rdm );

    else if (M == R && N == J && P == S)
      contribution += 1.0*getRDM(I,M,N,P,K,L,R,S, rdm );

    else if (M == R && N == J && P == I)
      contribution += 1.0*getRDM(M,N,P,S,K,L,R,S, rdm );

    else if (M == R && N == I && K == S)
      contribution += -1.0*getRDM(J,M,N,P,L,P,R,S, rdm );

    else if (M == R && N == I && P == S)
      contribution += -1.0*getRDM(J,M,N,P,K,L,R,S, rdm );

    else if (M == R && N == I && P == J)
      contribution += -1.0*getRDM(M,N,P,S,K,L,R,S, rdm );

    else if (M == R && N == P && K == S)
      contribution += -1.0*getRDM(I,J,M,N,L,P,R,S, rdm );

    else if (M == R && N == P && P == S)
      contribution += -1.0*getRDM(I,J,M,N,K,L,R,S, rdm );

    else if (M == R && N == P && P == J)
      contribution += 1.0*getRDM(I,M,N,S,K,L,R,S, rdm );

    else if (M == R && N == P && P == I)
      contribution += -1.0*getRDM(J,M,N,S,K,L,R,S, rdm );

    else if (M == J && K == S && L == R)
      contribution += 1.0*getRDM(I,M,N,P,N,P,R,S, rdm );

    else if (M == J && P == S && L == R)
      contribution += 1.0*getRDM(I,M,N,P,K,N,R,S, rdm );

    else if (M == J && P == R && K == S)
      contribution += 1.0*getRDM(I,M,N,P,L,N,R,S, rdm );

    else if (M == J && P == I && L == R)
      contribution += 1.0*getRDM(M,N,P,S,K,N,R,S, rdm );

    else if (M == J && P == I && K == S)
      contribution += 1.0*getRDM(M,N,P,R,L,N,R,S, rdm );

    else if (M == J && N == S && L == R)
      contribution += -1.0*getRDM(I,M,N,P,K,P,R,S, rdm );

    else if (M == J && N == S && P == R)
      contribution += 1.0*getRDM(I,M,N,P,K,L,R,S, rdm );

    else if (M == J && N == S && P == I)
      contribution += 1.0*getRDM(M,N,P,R,K,L,R,S, rdm );

    else if (M == J && N == R && K == S)
      contribution += -1.0*getRDM(I,M,N,P,L,P,R,S, rdm );

    else if (M == J && N == R && P == S)
      contribution += -1.0*getRDM(I,M,N,P,K,L,R,S, rdm );

    else if (M == J && N == R && P == I)
      contribution += -1.0*getRDM(M,N,P,S,K,L,R,S, rdm );

    else if (M == J && N == I && L == R)
      contribution += -1.0*getRDM(M,N,P,S,K,P,R,S, rdm );

    else if (M == J && N == I && K == S)
      contribution += -1.0*getRDM(M,N,P,R,L,P,R,S, rdm );

    else if (M == J && N == I && P == S)
      contribution += -1.0*getRDM(M,N,P,R,K,L,R,S, rdm );

    else if (M == J && N == I && P == R)
      contribution += 1.0*getRDM(M,N,P,S,K,L,R,S, rdm );

    else if (M == J && N == P && L == R)
      contribution += 1.0*getRDM(I,M,N,S,K,P,R,S, rdm );

    else if (M == J && N == P && K == S)
      contribution += 1.0*getRDM(I,M,N,R,L,P,R,S, rdm );

    else if (M == J && N == P && P == S)
      contribution += 1.0*getRDM(I,M,N,R,K,L,R,S, rdm );

    else if (M == J && N == P && P == R)
      contribution += -1.0*getRDM(I,M,N,S,K,L,R,S, rdm );

    else if (M == J && N == P && P == I)
      contribution += 1.0*getRDM(M,N,R,S,K,L,R,S, rdm );

    else if (M == I && K == S && L == R)
      contribution += -1.0*getRDM(J,M,N,P,N,P,R,S, rdm );

    else if (M == I && P == S && L == R)
      contribution += -1.0*getRDM(J,M,N,P,K,N,R,S, rdm );

    else if (M == I && P == R && K == S)
      contribution += -1.0*getRDM(J,M,N,P,L,N,R,S, rdm );

    else if (M == I && P == J && L == R)
      contribution += -1.0*getRDM(M,N,P,S,K,N,R,S, rdm );

    else if (M == I && P == J && K == S)
      contribution += -1.0*getRDM(M,N,P,R,L,N,R,S, rdm );

    else if (M == I && N == S && L == R)
      contribution += 1.0*getRDM(J,M,N,P,K,P,R,S, rdm );

    else if (M == I && N == S && P == R)
      contribution += -1.0*getRDM(J,M,N,P,K,L,R,S, rdm );

    else if (M == I && N == S && P == J)
      contribution += -1.0*getRDM(M,N,P,R,K,L,R,S, rdm );

    else if (M == I && N == R && K == S)
      contribution += 1.0*getRDM(J,M,N,P,L,P,R,S, rdm );

    else if (M == I && N == R && P == S)
      contribution += 1.0*getRDM(J,M,N,P,K,L,R,S, rdm );

    else if (M == I && N == R && P == J)
      contribution += 1.0*getRDM(M,N,P,S,K,L,R,S, rdm );

    else if (M == I && N == J && L == R)
      contribution += 1.0*getRDM(M,N,P,S,K,P,R,S, rdm );

    else if (M == I && N == J && K == S)
      contribution += 1.0*getRDM(M,N,P,R,L,P,R,S, rdm );

    else if (M == I && N == J && P == S)
      contribution += 1.0*getRDM(M,N,P,R,K,L,R,S, rdm );

    else if (M == I && N == J && P == R)
      contribution += -1.0*getRDM(M,N,P,S,K,L,R,S, rdm );

    else if (M == I && N == P && L == R)
      contribution += -1.0*getRDM(J,M,N,S,K,P,R,S, rdm );

    else if (M == I && N == P && K == S)
      contribution += -1.0*getRDM(J,M,N,R,L,P,R,S, rdm );

    else if (M == I && N == P && P == S)
      contribution += -1.0*getRDM(J,M,N,R,K,L,R,S, rdm );

    else if (M == I && N == P && P == R)
      contribution += 1.0*getRDM(J,M,N,S,K,L,R,S, rdm );

    else if (M == I && N == P && P == J)
      contribution += -1.0*getRDM(M,N,R,S,K,L,R,S, rdm );

    else if (M == P && K == S && L == R)
      contribution += -1.0*getRDM(I,J,M,N,N,P,R,S, rdm );

    else if (M == P && P == S && L == R)
      contribution += -1.0*getRDM(I,J,M,N,K,N,R,S, rdm );

    else if (M == P && P == R && K == S)
      contribution += -1.0*getRDM(I,J,M,N,L,N,R,S, rdm );

    else if (M == P && P == J && L == R)
      contribution += 1.0*getRDM(I,M,N,S,K,N,R,S, rdm );

    else if (M == P && P == J && K == S)
      contribution += 1.0*getRDM(I,M,N,R,L,N,R,S, rdm );

    else if (M == P && P == I && L == R)
      contribution += -1.0*getRDM(J,M,N,S,K,N,R,S, rdm );

    else if (M == P && P == I && K == S)
      contribution += -1.0*getRDM(J,M,N,R,L,N,R,S, rdm );

    else if (M == P && N == S && L == R)
      contribution += 1.0*getRDM(I,J,M,N,K,P,R,S, rdm );

    else if (M == P && N == S && P == R)
      contribution += -1.0*getRDM(I,J,M,N,K,L,R,S, rdm );

    else if (M == P && N == S && P == J)
      contribution += 1.0*getRDM(I,M,N,R,K,L,R,S, rdm );

    else if (M == P && N == S && P == I)
      contribution += -1.0*getRDM(J,M,N,R,K,L,R,S, rdm );

    else if (M == P && N == R && K == S)
      contribution += 1.0*getRDM(I,J,M,N,L,P,R,S, rdm );

    else if (M == P && N == R && P == S)
      contribution += 1.0*getRDM(I,J,M,N,K,L,R,S, rdm );

    else if (M == P && N == R && P == J)
      contribution += -1.0*getRDM(I,M,N,S,K,L,R,S, rdm );

    else if (M == P && N == R && P == I)
      contribution += 1.0*getRDM(J,M,N,S,K,L,R,S, rdm );

    else if (M == P && N == J && L == R)
      contribution += -1.0*getRDM(I,M,N,S,K,P,R,S, rdm );

    else if (M == P && N == J && K == S)
      contribution += -1.0*getRDM(I,M,N,R,L,P,R,S, rdm );

    else if (M == P && N == J && P == S)
      contribution += -1.0*getRDM(I,M,N,R,K,L,R,S, rdm );

    else if (M == P && N == J && P == R)
      contribution += 1.0*getRDM(I,M,N,S,K,L,R,S, rdm );

    else if (M == P && N == J && P == I)
      contribution += -1.0*getRDM(M,N,R,S,K,L,R,S, rdm );

    else if (M == P && N == I && L == R)
      contribution += 1.0*getRDM(J,M,N,S,K,P,R,S, rdm );

    else if (M == P && N == I && K == S)
      contribution += 1.0*getRDM(J,M,N,R,L,P,R,S, rdm );

    else if (M == P && N == I && P == S)
      contribution += 1.0*getRDM(J,M,N,R,K,L,R,S, rdm );

    else if (M == P && N == I && P == R)
      contribution += -1.0*getRDM(J,M,N,S,K,L,R,S, rdm );

    else if (M == P && N == I && P == J)
      contribution += 1.0*getRDM(M,N,R,S,K,L,R,S, rdm );

    else if (K == S && L == R)
      contribution += 1.0*getRDM(I,J,M,N,P,M,N,P,R,S, rdm );

    else if (P == S && L == R)
      contribution += -1.0*getRDM(I,J,M,N,P,K,M,N,R,S, rdm );

    else if (P == R && K == S)
      contribution += -1.0*getRDM(I,J,M,N,P,L,M,N,R,S, rdm );

    else if (P == J && L == R)
      contribution += -1.0*getRDM(I,M,N,P,S,K,M,N,R,S, rdm );

    else if (P == J && K == S)
      contribution += -1.0*getRDM(I,M,N,P,R,L,M,N,R,S, rdm );

    else if (P == I && L == R)
      contribution += 1.0*getRDM(J,M,N,P,S,K,M,N,R,S, rdm );

    else if (P == I && K == S)
      contribution += 1.0*getRDM(J,M,N,P,R,L,M,N,R,S, rdm );

    else if (N == S && L == R)
      contribution += 1.0*getRDM(I,J,M,N,P,K,M,P,R,S, rdm );

    else if (N == S && P == R)
      contribution += 1.0*getRDM(I,J,M,N,P,K,L,M,R,S, rdm );

    else if (N == S && P == J)
      contribution += 1.0*getRDM(I,M,N,P,R,K,L,M,R,S, rdm );

    else if (N == S && P == I)
      contribution += -1.0*getRDM(J,M,N,P,R,K,L,M,R,S, rdm );

    else if (N == R && K == S)
      contribution += 1.0*getRDM(I,J,M,N,P,L,M,P,R,S, rdm );

    else if (N == R && P == S)
      contribution += -1.0*getRDM(I,J,M,N,P,K,L,M,R,S, rdm );

    else if (N == R && P == J)
      contribution += -1.0*getRDM(I,M,N,P,S,K,L,M,R,S, rdm );

    else if (N == R && P == I)
      contribution += 1.0*getRDM(J,M,N,P,S,K,L,M,R,S, rdm );

    else if (N == J && L == R)
      contribution += 1.0*getRDM(I,M,N,P,S,K,M,P,R,S, rdm );

    else if (N == J && K == S)
      contribution += 1.0*getRDM(I,M,N,P,R,L,M,P,R,S, rdm );

    else if (N == J && P == S)
      contribution += -1.0*getRDM(I,M,N,P,R,K,L,M,R,S, rdm );

    else if (N == J && P == R)
      contribution += 1.0*getRDM(I,M,N,P,S,K,L,M,R,S, rdm );

    else if (N == J && P == I)
      contribution += 1.0*getRDM(M,N,P,R,S,K,L,M,R,S, rdm );

    else if (N == I && L == R)
      contribution += -1.0*getRDM(J,M,N,P,S,K,M,P,R,S, rdm );

    else if (N == I && K == S)
      contribution += -1.0*getRDM(J,M,N,P,R,L,M,P,R,S, rdm );

    else if (N == I && P == S)
      contribution += 1.0*getRDM(J,M,N,P,R,K,L,M,R,S, rdm );

    else if (N == I && P == R)
      contribution += -1.0*getRDM(J,M,N,P,S,K,L,M,R,S, rdm );

    else if (N == I && P == J)
      contribution += -1.0*getRDM(M,N,P,R,S,K,L,M,R,S, rdm );

    else if (N == P && L == R)
      contribution += -1.0*getRDM(I,J,M,N,S,K,M,P,R,S, rdm );

    else if (N == P && K == S)
      contribution += -1.0*getRDM(I,J,M,N,R,L,M,P,R,S, rdm );

    else if (N == P && P == S)
      contribution += 1.0*getRDM(I,J,M,N,R,K,L,M,R,S, rdm );

    else if (N == P && P == R)
      contribution += -1.0*getRDM(I,J,M,N,S,K,L,M,R,S, rdm );

    else if (N == P && P == J)
      contribution += 1.0*getRDM(I,M,N,R,S,K,L,M,R,S, rdm );

    else if (N == P && P == I)
      contribution += -1.0*getRDM(J,M,N,R,S,K,L,M,R,S, rdm );

    else if (M == S && L == R)
      contribution += -1.0*getRDM(I,J,M,N,P,K,N,P,R,S, rdm );

    else if (M == S && P == R)
      contribution += -1.0*getRDM(I,J,M,N,P,K,L,N,R,S, rdm );

    else if (M == S && P == J)
      contribution += -1.0*getRDM(I,M,N,P,R,K,L,N,R,S, rdm );

    else if (M == S && P == I)
      contribution += 1.0*getRDM(J,M,N,P,R,K,L,N,R,S, rdm );

    else if (M == S && N == R)
      contribution += 1.0*getRDM(I,J,M,N,P,K,L,P,R,S, rdm );

    else if (M == S && N == J)
      contribution += 1.0*getRDM(I,M,N,P,R,K,L,P,R,S, rdm );

    else if (M == S && N == I)
      contribution += -1.0*getRDM(J,M,N,P,R,K,L,P,R,S, rdm );

    else if (M == S && N == P)
      contribution += -1.0*getRDM(I,J,M,N,R,K,L,P,R,S, rdm );

    else if (M == R && K == S)
      contribution += -1.0*getRDM(I,J,M,N,P,L,N,P,R,S, rdm );

    else if (M == R && P == S)
      contribution += 1.0*getRDM(I,J,M,N,P,K,L,N,R,S, rdm );

    else if (M == R && P == J)
      contribution += 1.0*getRDM(I,M,N,P,S,K,L,N,R,S, rdm );

    else if (M == R && P == I)
      contribution += -1.0*getRDM(J,M,N,P,S,K,L,N,R,S, rdm );

    else if (M == R && N == S)
      contribution += -1.0*getRDM(I,J,M,N,P,K,L,P,R,S, rdm );

    else if (M == R && N == J)
      contribution += -1.0*getRDM(I,M,N,P,S,K,L,P,R,S, rdm );

    else if (M == R && N == I)
      contribution += 1.0*getRDM(J,M,N,P,S,K,L,P,R,S, rdm );

    else if (M == R && N == P)
      contribution += 1.0*getRDM(I,J,M,N,S,K,L,P,R,S, rdm );

    else if (M == J && L == R)
      contribution += -1.0*getRDM(I,M,N,P,S,K,N,P,R,S, rdm );

    else if (M == J && K == S)
      contribution += -1.0*getRDM(I,M,N,P,R,L,N,P,R,S, rdm );

    else if (M == J && P == S)
      contribution += 1.0*getRDM(I,M,N,P,R,K,L,N,R,S, rdm );

    else if (M == J && P == R)
      contribution += -1.0*getRDM(I,M,N,P,S,K,L,N,R,S, rdm );

    else if (M == J && P == I)
      contribution += -1.0*getRDM(M,N,P,R,S,K,L,N,R,S, rdm );

    else if (M == J && N == S)
      contribution += -1.0*getRDM(I,M,N,P,R,K,L,P,R,S, rdm );

    else if (M == J && N == R)
      contribution += 1.0*getRDM(I,M,N,P,S,K,L,P,R,S, rdm );

    else if (M == J && N == I)
      contribution += 1.0*getRDM(M,N,P,R,S,K,L,P,R,S, rdm );

    else if (M == J && N == P)
      contribution += -1.0*getRDM(I,M,N,R,S,K,L,P,R,S, rdm );

    else if (M == I && L == R)
      contribution += 1.0*getRDM(J,M,N,P,S,K,N,P,R,S, rdm );

    else if (M == I && K == S)
      contribution += 1.0*getRDM(J,M,N,P,R,L,N,P,R,S, rdm );

    else if (M == I && P == S)
      contribution += -1.0*getRDM(J,M,N,P,R,K,L,N,R,S, rdm );

    else if (M == I && P == R)
      contribution += 1.0*getRDM(J,M,N,P,S,K,L,N,R,S, rdm );

    else if (M == I && P == J)
      contribution += 1.0*getRDM(M,N,P,R,S,K,L,N,R,S, rdm );

    else if (M == I && N == S)
      contribution += 1.0*getRDM(J,M,N,P,R,K,L,P,R,S, rdm );

    else if (M == I && N == R)
      contribution += -1.0*getRDM(J,M,N,P,S,K,L,P,R,S, rdm );

    else if (M == I && N == J)
      contribution += -1.0*getRDM(M,N,P,R,S,K,L,P,R,S, rdm );

    else if (M == I && N == P)
      contribution += 1.0*getRDM(J,M,N,R,S,K,L,P,R,S, rdm );

    else if (M == P && L == R)
      contribution += 1.0*getRDM(I,J,M,N,S,K,N,P,R,S, rdm );

    else if (M == P && K == S)
      contribution += 1.0*getRDM(I,J,M,N,R,L,N,P,R,S, rdm );

    else if (M == P && P == S)
      contribution += -1.0*getRDM(I,J,M,N,R,K,L,N,R,S, rdm );

    else if (M == P && P == R)
      contribution += 1.0*getRDM(I,J,M,N,S,K,L,N,R,S, rdm );

    else if (M == P && P == J)
      contribution += -1.0*getRDM(I,M,N,R,S,K,L,N,R,S, rdm );

    else if (M == P && P == I)
      contribution += 1.0*getRDM(J,M,N,R,S,K,L,N,R,S, rdm );

    else if (M == P && N == S)
      contribution += 1.0*getRDM(I,J,M,N,R,K,L,P,R,S, rdm );

    else if (M == P && N == R)
      contribution += -1.0*getRDM(I,J,M,N,S,K,L,P,R,S, rdm );

    else if (M == P && N == J)
      contribution += 1.0*getRDM(I,M,N,R,S,K,L,P,R,S, rdm );

    else if (M == P && N == I)
      contribution += -1.0*getRDM(J,M,N,R,S,K,L,P,R,S, rdm );

    else if (L == R)
      contribution += -1.0*getRDM(I,J,M,N,P,S,K,M,N,P,R,S, rdm );

    else if (K == S)
      contribution += -1.0*getRDM(I,J,M,N,P,R,L,M,N,P,R,S, rdm );

    else if (P == S)
      contribution += -1.0*getRDM(I,J,M,N,P,R,K,L,M,N,R,S, rdm );

    else if (P == R)
      contribution += 1.0*getRDM(I,J,M,N,P,S,K,L,M,N,R,S, rdm );

    else if (P == J)
      contribution += 1.0*getRDM(I,M,N,P,R,S,K,L,M,N,R,S, rdm );

    else if (P == I)
      contribution += -1.0*getRDM(J,M,N,P,R,S,K,L,M,N,R,S, rdm );

    else if (N == S)
      contribution += 1.0*getRDM(I,J,M,N,P,R,K,L,M,P,R,S, rdm );

    else if (N == R)
      contribution += -1.0*getRDM(I,J,M,N,P,S,K,L,M,P,R,S, rdm );

    else if (N == J)
      contribution += -1.0*getRDM(I,M,N,P,R,S,K,L,M,P,R,S, rdm );

    else if (N == I)
      contribution += 1.0*getRDM(J,M,N,P,R,S,K,L,M,P,R,S, rdm );

    else if (N == P)
      contribution += 1.0*getRDM(I,J,M,N,R,S,K,L,M,P,R,S, rdm );

    else if (M == S)
      contribution += -1.0*getRDM(I,J,M,N,P,R,K,L,N,P,R,S, rdm );

    else if (M == R)
      contribution += 1.0*getRDM(I,J,M,N,P,S,K,L,N,P,R,S, rdm );

    else if (M == J)
      contribution += 1.0*getRDM(I,M,N,P,R,S,K,L,N,P,R,S, rdm );

    else if (M == I)
      contribution += -1.0*getRDM(J,M,N,P,R,S,K,L,N,P,R,S, rdm );

    else if (M == P)
      contribution += -1.0*getRDM(I,J,M,N,R,S,K,L,N,P,R,S, rdm );

    else
      contribution += 1.0*getRDM(I,J,M,N,P,R,S,K,L,M,N,P,R,S, rdm );

    return contribution;
    
  }
  
  complexT calcTermGJ2n(int M, int N, int P, int Q, int I, int J, int K, int L, int R, int S, MatrixXcT& rdm) {
    complexT contribution = complexT(0.,0.);
    if (M == Q && N == P && P == J && Q == I && K == S && L == R)
      contribution += -1.0*getRDM(M,N,R,S, rdm );

    else if (M == P && N == Q && P == J && Q == I && K == S && L == R)
      contribution += 1.0*getRDM(M,N,R,S, rdm );

    else if (N == Q && P == J && Q == I && K == S && L == R)
      contribution += -1.0*getRDM(M,N,P,M,R,S, rdm );

    else if (N == P && P == J && Q == I && K == S && L == R)
      contribution += 1.0*getRDM(M,N,Q,M,R,S, rdm );

    else if (M == S && N == Q && P == J && Q == I && L == R)
      contribution += 1.0*getRDM(M,N,P,K,R,S, rdm );

    else if (M == S && N == P && P == J && Q == I && L == R)
      contribution += -1.0*getRDM(M,N,Q,K,R,S, rdm );

    else if (M == R && N == Q && P == J && Q == I && K == S)
      contribution += 1.0*getRDM(M,N,P,L,R,S, rdm );

    else if (M == R && N == P && P == J && Q == I && K == S)
      contribution += -1.0*getRDM(M,N,Q,L,R,S, rdm );

    else if (M == J && N == Q && Q == I && K == S && L == R)
      contribution += 1.0*getRDM(M,N,P,P,R,S, rdm );

    else if (M == J && N == Q && P == S && Q == I && L == R)
      contribution += -1.0*getRDM(M,N,P,K,R,S, rdm );

    else if (M == J && N == Q && P == R && Q == I && K == S)
      contribution += -1.0*getRDM(M,N,P,L,R,S, rdm );

    else if (M == J && N == P && Q == I && K == S && L == R)
      contribution += -1.0*getRDM(M,N,Q,P,R,S, rdm );

    else if (M == J && N == P && P == S && Q == I && L == R)
      contribution += 1.0*getRDM(M,N,Q,K,R,S, rdm );

    else if (M == J && N == P && P == R && Q == I && K == S)
      contribution += 1.0*getRDM(M,N,Q,L,R,S, rdm );

    else if (M == I && N == Q && P == J && K == S && L == R)
      contribution += 1.0*getRDM(M,N,P,Q,R,S, rdm );

    else if (M == I && N == Q && P == J && Q == S && L == R)
      contribution += -1.0*getRDM(M,N,P,K,R,S, rdm );

    else if (M == I && N == Q && P == J && Q == R && K == S)
      contribution += -1.0*getRDM(M,N,P,L,R,S, rdm );

    else if (M == I && N == P && P == J && K == S && L == R)
      contribution += -1.0*getRDM(M,N,Q,Q,R,S, rdm );

    else if (M == I && N == P && P == J && Q == S && L == R)
      contribution += 1.0*getRDM(M,N,Q,K,R,S, rdm );

    else if (M == I && N == P && P == J && Q == R && K == S)
      contribution += 1.0*getRDM(M,N,Q,L,R,S, rdm );

    else if (M == Q && P == J && Q == I && K == S && L == R)
      contribution += 1.0*getRDM(M,N,P,N,R,S, rdm );

    else if (M == Q && N == S && P == J && Q == I && L == R)
      contribution += -1.0*getRDM(M,N,P,K,R,S, rdm );

    else if (M == Q && N == R && P == J && Q == I && K == S)
      contribution += -1.0*getRDM(M,N,P,L,R,S, rdm );

    else if (M == Q && N == J && Q == I && K == S && L == R)
      contribution += -1.0*getRDM(M,N,P,P,R,S, rdm );

    else if (M == Q && N == J && P == S && Q == I && L == R)
      contribution += 1.0*getRDM(M,N,P,K,R,S, rdm );

    else if (M == Q && N == J && P == R && Q == I && K == S)
      contribution += 1.0*getRDM(M,N,P,L,R,S, rdm );

    else if (M == Q && N == I && P == J && K == S && L == R)
      contribution += -1.0*getRDM(M,N,P,Q,R,S, rdm );

    else if (M == Q && N == I && P == J && Q == S && L == R)
      contribution += 1.0*getRDM(M,N,P,K,R,S, rdm );

    else if (M == Q && N == I && P == J && Q == R && K == S)
      contribution += 1.0*getRDM(M,N,P,L,R,S, rdm );

    else if (M == Q && N == P && Q == I && K == S && L == R)
      contribution += 1.0*getRDM(J,M,N,P,R,S, rdm );

    else if (M == Q && N == P && P == S && Q == I && L == R)
      contribution += -1.0*getRDM(J,M,N,K,R,S, rdm );

    else if (M == Q && N == P && P == R && Q == I && K == S)
      contribution += -1.0*getRDM(J,M,N,L,R,S, rdm );

    else if (M == Q && N == P && P == J && K == S && L == R)
      contribution += 1.0*getRDM(I,M,N,Q,R,S, rdm );

    else if (M == Q && N == P && P == J && Q == S && L == R)
      contribution += -1.0*getRDM(I,M,N,K,R,S, rdm );

    else if (M == Q && N == P && P == J && Q == R && K == S)
      contribution += -1.0*getRDM(I,M,N,L,R,S, rdm );

    else if (M == Q && N == P && P == J && Q == I && L == R)
      contribution += 1.0*getRDM(M,N,S,K,R,S, rdm );

    else if (M == Q && N == P && P == J && Q == I && K == S)
      contribution += 1.0*getRDM(M,N,R,L,R,S, rdm );

    else if (M == P && P == J && Q == I && K == S && L == R)
      contribution += -1.0*getRDM(M,N,Q,N,R,S, rdm );

    else if (M == P && N == S && P == J && Q == I && L == R)
      contribution += 1.0*getRDM(M,N,Q,K,R,S, rdm );

    else if (M == P && N == R && P == J && Q == I && K == S)
      contribution += 1.0*getRDM(M,N,Q,L,R,S, rdm );

    else if (M == P && N == J && Q == I && K == S && L == R)
      contribution += 1.0*getRDM(M,N,Q,P,R,S, rdm );

    else if (M == P && N == J && P == S && Q == I && L == R)
      contribution += -1.0*getRDM(M,N,Q,K,R,S, rdm );

    else if (M == P && N == J && P == R && Q == I && K == S)
      contribution += -1.0*getRDM(M,N,Q,L,R,S, rdm );

    else if (M == P && N == I && P == J && K == S && L == R)
      contribution += 1.0*getRDM(M,N,Q,Q,R,S, rdm );

    else if (M == P && N == I && P == J && Q == S && L == R)
      contribution += -1.0*getRDM(M,N,Q,K,R,S, rdm );

    else if (M == P && N == I && P == J && Q == R && K == S)
      contribution += -1.0*getRDM(M,N,Q,L,R,S, rdm );

    else if (M == P && N == Q && Q == I && K == S && L == R)
      contribution += -1.0*getRDM(J,M,N,P,R,S, rdm );

    else if (M == P && N == Q && P == S && Q == I && L == R)
      contribution += 1.0*getRDM(J,M,N,K,R,S, rdm );

    else if (M == P && N == Q && P == R && Q == I && K == S)
      contribution += 1.0*getRDM(J,M,N,L,R,S, rdm );

    else if (M == P && N == Q && P == J && K == S && L == R)
      contribution += -1.0*getRDM(I,M,N,Q,R,S, rdm );

    else if (M == P && N == Q && P == J && Q == S && L == R)
      contribution += 1.0*getRDM(I,M,N,K,R,S, rdm );

    else if (M == P && N == Q && P == J && Q == R && K == S)
      contribution += 1.0*getRDM(I,M,N,L,R,S, rdm );

    else if (M == P && N == Q && P == J && Q == I && L == R)
      contribution += -1.0*getRDM(M,N,S,K,R,S, rdm );

    else if (M == P && N == Q && P == J && Q == I && K == S)
      contribution += -1.0*getRDM(M,N,R,L,R,S, rdm );

    else if (P == J && Q == I && K == S && L == R)
      contribution += -1.0*getRDM(M,N,P,Q,M,N,R,S, rdm );

    else if (N == S && P == J && Q == I && L == R)
      contribution += -1.0*getRDM(M,N,P,Q,K,M,R,S, rdm );

    else if (N == R && P == J && Q == I && K == S)
      contribution += -1.0*getRDM(M,N,P,Q,L,M,R,S, rdm );

    else if (N == J && Q == I && K == S && L == R)
      contribution += 1.0*getRDM(M,N,P,Q,M,P,R,S, rdm );

    else if (N == J && P == S && Q == I && L == R)
      contribution += 1.0*getRDM(M,N,P,Q,K,M,R,S, rdm );

    else if (N == J && P == R && Q == I && K == S)
      contribution += 1.0*getRDM(M,N,P,Q,L,M,R,S, rdm );

    else if (N == I && P == J && K == S && L == R)
      contribution += 1.0*getRDM(M,N,P,Q,M,Q,R,S, rdm );

    else if (N == I && P == J && Q == S && L == R)
      contribution += 1.0*getRDM(M,N,P,Q,K,M,R,S, rdm );

    else if (N == I && P == J && Q == R && K == S)
      contribution += 1.0*getRDM(M,N,P,Q,L,M,R,S, rdm );

    else if (N == Q && Q == I && K == S && L == R)
      contribution += 1.0*getRDM(J,M,N,P,M,P,R,S, rdm );

    else if (N == Q && P == S && Q == I && L == R)
      contribution += 1.0*getRDM(J,M,N,P,K,M,R,S, rdm );

    else if (N == Q && P == R && Q == I && K == S)
      contribution += 1.0*getRDM(J,M,N,P,L,M,R,S, rdm );

    else if (N == Q && P == J && K == S && L == R)
      contribution += 1.0*getRDM(I,M,N,P,M,Q,R,S, rdm );

    else if (N == Q && P == J && Q == S && L == R)
      contribution += 1.0*getRDM(I,M,N,P,K,M,R,S, rdm );

    else if (N == Q && P == J && Q == R && K == S)
      contribution += 1.0*getRDM(I,M,N,P,L,M,R,S, rdm );

    else if (N == Q && P == J && Q == I && L == R)
      contribution += 1.0*getRDM(M,N,P,S,K,M,R,S, rdm );

    else if (N == Q && P == J && Q == I && K == S)
      contribution += 1.0*getRDM(M,N,P,R,L,M,R,S, rdm );

    else if (N == P && Q == I && K == S && L == R)
      contribution += -1.0*getRDM(J,M,N,Q,M,P,R,S, rdm );

    else if (N == P && P == S && Q == I && L == R)
      contribution += -1.0*getRDM(J,M,N,Q,K,M,R,S, rdm );

    else if (N == P && P == R && Q == I && K == S)
      contribution += -1.0*getRDM(J,M,N,Q,L,M,R,S, rdm );

    else if (N == P && P == J && K == S && L == R)
      contribution += -1.0*getRDM(I,M,N,Q,M,Q,R,S, rdm );

    else if (N == P && P == J && Q == S && L == R)
      contribution += -1.0*getRDM(I,M,N,Q,K,M,R,S, rdm );

    else if (N == P && P == J && Q == R && K == S)
      contribution += -1.0*getRDM(I,M,N,Q,L,M,R,S, rdm );

    else if (N == P && P == J && Q == I && L == R)
      contribution += -1.0*getRDM(M,N,Q,S,K,M,R,S, rdm );

    else if (N == P && P == J && Q == I && K == S)
      contribution += -1.0*getRDM(M,N,Q,R,L,M,R,S, rdm );

    else if (M == S && P == J && Q == I && L == R)
      contribution += 1.0*getRDM(M,N,P,Q,K,N,R,S, rdm );

    else if (M == S && N == R && P == J && Q == I)
      contribution += -1.0*getRDM(M,N,P,Q,K,L,R,S, rdm );

    else if (M == S && N == J && Q == I && L == R)
      contribution += -1.0*getRDM(M,N,P,Q,K,P,R,S, rdm );

    else if (M == S && N == J && P == R && Q == I)
      contribution += 1.0*getRDM(M,N,P,Q,K,L,R,S, rdm );

    else if (M == S && N == I && P == J && L == R)
      contribution += -1.0*getRDM(M,N,P,Q,K,Q,R,S, rdm );

    else if (M == S && N == I && P == J && Q == R)
      contribution += 1.0*getRDM(M,N,P,Q,K,L,R,S, rdm );

    else if (M == S && N == Q && Q == I && L == R)
      contribution += -1.0*getRDM(J,M,N,P,K,P,R,S, rdm );

    else if (M == S && N == Q && P == R && Q == I)
      contribution += 1.0*getRDM(J,M,N,P,K,L,R,S, rdm );

    else if (M == S && N == Q && P == J && L == R)
      contribution += -1.0*getRDM(I,M,N,P,K,Q,R,S, rdm );

    else if (M == S && N == Q && P == J && Q == R)
      contribution += 1.0*getRDM(I,M,N,P,K,L,R,S, rdm );

    else if (M == S && N == Q && P == J && Q == I)
      contribution += 1.0*getRDM(M,N,P,R,K,L,R,S, rdm );

    else if (M == S && N == P && Q == I && L == R)
      contribution += 1.0*getRDM(J,M,N,Q,K,P,R,S, rdm );

    else if (M == S && N == P && P == R && Q == I)
      contribution += -1.0*getRDM(J,M,N,Q,K,L,R,S, rdm );

    else if (M == S && N == P && P == J && L == R)
      contribution += 1.0*getRDM(I,M,N,Q,K,Q,R,S, rdm );

    else if (M == S && N == P && P == J && Q == R)
      contribution += -1.0*getRDM(I,M,N,Q,K,L,R,S, rdm );

    else if (M == S && N == P && P == J && Q == I)
      contribution += -1.0*getRDM(M,N,Q,R,K,L,R,S, rdm );

    else if (M == R && P == J && Q == I && K == S)
      contribution += 1.0*getRDM(M,N,P,Q,L,N,R,S, rdm );

    else if (M == R && N == S && P == J && Q == I)
      contribution += 1.0*getRDM(M,N,P,Q,K,L,R,S, rdm );

    else if (M == R && N == J && Q == I && K == S)
      contribution += -1.0*getRDM(M,N,P,Q,L,P,R,S, rdm );

    else if (M == R && N == J && P == S && Q == I)
      contribution += -1.0*getRDM(M,N,P,Q,K,L,R,S, rdm );

    else if (M == R && N == I && P == J && K == S)
      contribution += -1.0*getRDM(M,N,P,Q,L,Q,R,S, rdm );

    else if (M == R && N == I && P == J && Q == S)
      contribution += -1.0*getRDM(M,N,P,Q,K,L,R,S, rdm );

    else if (M == R && N == Q && Q == I && K == S)
      contribution += -1.0*getRDM(J,M,N,P,L,P,R,S, rdm );

    else if (M == R && N == Q && P == S && Q == I)
      contribution += -1.0*getRDM(J,M,N,P,K,L,R,S, rdm );

    else if (M == R && N == Q && P == J && K == S)
      contribution += -1.0*getRDM(I,M,N,P,L,Q,R,S, rdm );

    else if (M == R && N == Q && P == J && Q == S)
      contribution += -1.0*getRDM(I,M,N,P,K,L,R,S, rdm );

    else if (M == R && N == Q && P == J && Q == I)
      contribution += -1.0*getRDM(M,N,P,S,K,L,R,S, rdm );

    else if (M == R && N == P && Q == I && K == S)
      contribution += 1.0*getRDM(J,M,N,Q,L,P,R,S, rdm );

    else if (M == R && N == P && P == S && Q == I)
      contribution += 1.0*getRDM(J,M,N,Q,K,L,R,S, rdm );

    else if (M == R && N == P && P == J && K == S)
      contribution += 1.0*getRDM(I,M,N,Q,L,Q,R,S, rdm );

    else if (M == R && N == P && P == J && Q == S)
      contribution += 1.0*getRDM(I,M,N,Q,K,L,R,S, rdm );

    else if (M == R && N == P && P == J && Q == I)
      contribution += 1.0*getRDM(M,N,Q,S,K,L,R,S, rdm );

    else if (M == J && Q == I && K == S && L == R)
      contribution += -1.0*getRDM(M,N,P,Q,N,P,R,S, rdm );

    else if (M == J && P == S && Q == I && L == R)
      contribution += -1.0*getRDM(M,N,P,Q,K,N,R,S, rdm );

    else if (M == J && P == R && Q == I && K == S)
      contribution += -1.0*getRDM(M,N,P,Q,L,N,R,S, rdm );

    else if (M == J && N == S && Q == I && L == R)
      contribution += 1.0*getRDM(M,N,P,Q,K,P,R,S, rdm );

    else if (M == J && N == S && P == R && Q == I)
      contribution += -1.0*getRDM(M,N,P,Q,K,L,R,S, rdm );

    else if (M == J && N == R && Q == I && K == S)
      contribution += 1.0*getRDM(M,N,P,Q,L,P,R,S, rdm );

    else if (M == J && N == R && P == S && Q == I)
      contribution += 1.0*getRDM(M,N,P,Q,K,L,R,S, rdm );

    else if (M == J && N == I && K == S && L == R)
      contribution += -1.0*getRDM(M,N,P,Q,P,Q,R,S, rdm );

    else if (M == J && N == I && Q == S && L == R)
      contribution += -1.0*getRDM(M,N,P,Q,K,P,R,S, rdm );

    else if (M == J && N == I && Q == R && K == S)
      contribution += -1.0*getRDM(M,N,P,Q,L,P,R,S, rdm );

    else if (M == J && N == I && P == S && L == R)
      contribution += 1.0*getRDM(M,N,P,Q,K,Q,R,S, rdm );

    else if (M == J && N == I && P == S && Q == R)
      contribution += -1.0*getRDM(M,N,P,Q,K,L,R,S, rdm );

    else if (M == J && N == I && P == R && K == S)
      contribution += 1.0*getRDM(M,N,P,Q,L,Q,R,S, rdm );

    else if (M == J && N == I && P == R && Q == S)
      contribution += 1.0*getRDM(M,N,P,Q,K,L,R,S, rdm );

    else if (M == J && N == Q && K == S && L == R)
      contribution += -1.0*getRDM(I,M,N,P,P,Q,R,S, rdm );

    else if (M == J && N == Q && Q == S && L == R)
      contribution += -1.0*getRDM(I,M,N,P,K,P,R,S, rdm );

    else if (M == J && N == Q && Q == R && K == S)
      contribution += -1.0*getRDM(I,M,N,P,L,P,R,S, rdm );

    else if (M == J && N == Q && Q == I && L == R)
      contribution += -1.0*getRDM(M,N,P,S,K,P,R,S, rdm );

    else if (M == J && N == Q && Q == I && K == S)
      contribution += -1.0*getRDM(M,N,P,R,L,P,R,S, rdm );

    else if (M == J && N == Q && P == S && L == R)
      contribution += 1.0*getRDM(I,M,N,P,K,Q,R,S, rdm );

    else if (M == J && N == Q && P == S && Q == R)
      contribution += -1.0*getRDM(I,M,N,P,K,L,R,S, rdm );

    else if (M == J && N == Q && P == S && Q == I)
      contribution += -1.0*getRDM(M,N,P,R,K,L,R,S, rdm );

    else if (M == J && N == Q && P == R && K == S)
      contribution += 1.0*getRDM(I,M,N,P,L,Q,R,S, rdm );

    else if (M == J && N == Q && P == R && Q == S)
      contribution += 1.0*getRDM(I,M,N,P,K,L,R,S, rdm );

    else if (M == J && N == Q && P == R && Q == I)
      contribution += 1.0*getRDM(M,N,P,S,K,L,R,S, rdm );

    else if (M == J && N == P && K == S && L == R)
      contribution += 1.0*getRDM(I,M,N,Q,P,Q,R,S, rdm );

    else if (M == J && N == P && Q == S && L == R)
      contribution += 1.0*getRDM(I,M,N,Q,K,P,R,S, rdm );

    else if (M == J && N == P && Q == R && K == S)
      contribution += 1.0*getRDM(I,M,N,Q,L,P,R,S, rdm );

    else if (M == J && N == P && Q == I && L == R)
      contribution += 1.0*getRDM(M,N,Q,S,K,P,R,S, rdm );

    else if (M == J && N == P && Q == I && K == S)
      contribution += 1.0*getRDM(M,N,Q,R,L,P,R,S, rdm );

    else if (M == J && N == P && P == S && L == R)
      contribution += -1.0*getRDM(I,M,N,Q,K,Q,R,S, rdm );

    else if (M == J && N == P && P == S && Q == R)
      contribution += 1.0*getRDM(I,M,N,Q,K,L,R,S, rdm );

    else if (M == J && N == P && P == S && Q == I)
      contribution += 1.0*getRDM(M,N,Q,R,K,L,R,S, rdm );

    else if (M == J && N == P && P == R && K == S)
      contribution += -1.0*getRDM(I,M,N,Q,L,Q,R,S, rdm );

    else if (M == J && N == P && P == R && Q == S)
      contribution += -1.0*getRDM(I,M,N,Q,K,L,R,S, rdm );

    else if (M == J && N == P && P == R && Q == I)
      contribution += -1.0*getRDM(M,N,Q,S,K,L,R,S, rdm );

    else if (M == I && P == J && K == S && L == R)
      contribution += -1.0*getRDM(M,N,P,Q,N,Q,R,S, rdm );

    else if (M == I && P == J && Q == S && L == R)
      contribution += -1.0*getRDM(M,N,P,Q,K,N,R,S, rdm );

    else if (M == I && P == J && Q == R && K == S)
      contribution += -1.0*getRDM(M,N,P,Q,L,N,R,S, rdm );

    else if (M == I && N == S && P == J && L == R)
      contribution += 1.0*getRDM(M,N,P,Q,K,Q,R,S, rdm );

    else if (M == I && N == S && P == J && Q == R)
      contribution += -1.0*getRDM(M,N,P,Q,K,L,R,S, rdm );

    else if (M == I && N == R && P == J && K == S)
      contribution += 1.0*getRDM(M,N,P,Q,L,Q,R,S, rdm );

    else if (M == I && N == R && P == J && Q == S)
      contribution += 1.0*getRDM(M,N,P,Q,K,L,R,S, rdm );

    else if (M == I && N == J && K == S && L == R)
      contribution += 1.0*getRDM(M,N,P,Q,P,Q,R,S, rdm );

    else if (M == I && N == J && Q == S && L == R)
      contribution += 1.0*getRDM(M,N,P,Q,K,P,R,S, rdm );

    else if (M == I && N == J && Q == R && K == S)
      contribution += 1.0*getRDM(M,N,P,Q,L,P,R,S, rdm );

    else if (M == I && N == J && P == S && L == R)
      contribution += -1.0*getRDM(M,N,P,Q,K,Q,R,S, rdm );

    else if (M == I && N == J && P == S && Q == R)
      contribution += 1.0*getRDM(M,N,P,Q,K,L,R,S, rdm );

    else if (M == I && N == J && P == R && K == S)
      contribution += -1.0*getRDM(M,N,P,Q,L,Q,R,S, rdm );

    else if (M == I && N == J && P == R && Q == S)
      contribution += -1.0*getRDM(M,N,P,Q,K,L,R,S, rdm );

    else if (M == I && N == Q && K == S && L == R)
      contribution += 1.0*getRDM(J,M,N,P,P,Q,R,S, rdm );

    else if (M == I && N == Q && Q == S && L == R)
      contribution += 1.0*getRDM(J,M,N,P,K,P,R,S, rdm );

    else if (M == I && N == Q && Q == R && K == S)
      contribution += 1.0*getRDM(J,M,N,P,L,P,R,S, rdm );

    else if (M == I && N == Q && P == S && L == R)
      contribution += -1.0*getRDM(J,M,N,P,K,Q,R,S, rdm );

    else if (M == I && N == Q && P == S && Q == R)
      contribution += 1.0*getRDM(J,M,N,P,K,L,R,S, rdm );

    else if (M == I && N == Q && P == R && K == S)
      contribution += -1.0*getRDM(J,M,N,P,L,Q,R,S, rdm );

    else if (M == I && N == Q && P == R && Q == S)
      contribution += -1.0*getRDM(J,M,N,P,K,L,R,S, rdm );

    else if (M == I && N == Q && P == J && L == R)
      contribution += -1.0*getRDM(M,N,P,S,K,Q,R,S, rdm );

    else if (M == I && N == Q && P == J && K == S)
      contribution += -1.0*getRDM(M,N,P,R,L,Q,R,S, rdm );

    else if (M == I && N == Q && P == J && Q == S)
      contribution += -1.0*getRDM(M,N,P,R,K,L,R,S, rdm );

    else if (M == I && N == Q && P == J && Q == R)
      contribution += 1.0*getRDM(M,N,P,S,K,L,R,S, rdm );

    else if (M == I && N == P && K == S && L == R)
      contribution += -1.0*getRDM(J,M,N,Q,P,Q,R,S, rdm );

    else if (M == I && N == P && Q == S && L == R)
      contribution += -1.0*getRDM(J,M,N,Q,K,P,R,S, rdm );

    else if (M == I && N == P && Q == R && K == S)
      contribution += -1.0*getRDM(J,M,N,Q,L,P,R,S, rdm );

    else if (M == I && N == P && P == S && L == R)
      contribution += 1.0*getRDM(J,M,N,Q,K,Q,R,S, rdm );

    else if (M == I && N == P && P == S && Q == R)
      contribution += -1.0*getRDM(J,M,N,Q,K,L,R,S, rdm );

    else if (M == I && N == P && P == R && K == S)
      contribution += 1.0*getRDM(J,M,N,Q,L,Q,R,S, rdm );

    else if (M == I && N == P && P == R && Q == S)
      contribution += 1.0*getRDM(J,M,N,Q,K,L,R,S, rdm );

    else if (M == I && N == P && P == J && L == R)
      contribution += 1.0*getRDM(M,N,Q,S,K,Q,R,S, rdm );

    else if (M == I && N == P && P == J && K == S)
      contribution += 1.0*getRDM(M,N,Q,R,L,Q,R,S, rdm );

    else if (M == I && N == P && P == J && Q == S)
      contribution += 1.0*getRDM(M,N,Q,R,K,L,R,S, rdm );

    else if (M == I && N == P && P == J && Q == R)
      contribution += -1.0*getRDM(M,N,Q,S,K,L,R,S, rdm );

    else if (M == Q && Q == I && K == S && L == R)
      contribution += -1.0*getRDM(J,M,N,P,N,P,R,S, rdm );

    else if (M == Q && P == S && Q == I && L == R)
      contribution += -1.0*getRDM(J,M,N,P,K,N,R,S, rdm );

    else if (M == Q && P == R && Q == I && K == S)
      contribution += -1.0*getRDM(J,M,N,P,L,N,R,S, rdm );

    else if (M == Q && P == J && K == S && L == R)
      contribution += -1.0*getRDM(I,M,N,P,N,Q,R,S, rdm );

    else if (M == Q && P == J && Q == S && L == R)
      contribution += -1.0*getRDM(I,M,N,P,K,N,R,S, rdm );

    else if (M == Q && P == J && Q == R && K == S)
      contribution += -1.0*getRDM(I,M,N,P,L,N,R,S, rdm );

    else if (M == Q && P == J && Q == I && L == R)
      contribution += -1.0*getRDM(M,N,P,S,K,N,R,S, rdm );

    else if (M == Q && P == J && Q == I && K == S)
      contribution += -1.0*getRDM(M,N,P,R,L,N,R,S, rdm );

    else if (M == Q && N == S && Q == I && L == R)
      contribution += 1.0*getRDM(J,M,N,P,K,P,R,S, rdm );

    else if (M == Q && N == S && P == R && Q == I)
      contribution += -1.0*getRDM(J,M,N,P,K,L,R,S, rdm );

    else if (M == Q && N == S && P == J && L == R)
      contribution += 1.0*getRDM(I,M,N,P,K,Q,R,S, rdm );

    else if (M == Q && N == S && P == J && Q == R)
      contribution += -1.0*getRDM(I,M,N,P,K,L,R,S, rdm );

    else if (M == Q && N == S && P == J && Q == I)
      contribution += -1.0*getRDM(M,N,P,R,K,L,R,S, rdm );

    else if (M == Q && N == R && Q == I && K == S)
      contribution += 1.0*getRDM(J,M,N,P,L,P,R,S, rdm );

    else if (M == Q && N == R && P == S && Q == I)
      contribution += 1.0*getRDM(J,M,N,P,K,L,R,S, rdm );

    else if (M == Q && N == R && P == J && K == S)
      contribution += 1.0*getRDM(I,M,N,P,L,Q,R,S, rdm );

    else if (M == Q && N == R && P == J && Q == S)
      contribution += 1.0*getRDM(I,M,N,P,K,L,R,S, rdm );

    else if (M == Q && N == R && P == J && Q == I)
      contribution += 1.0*getRDM(M,N,P,S,K,L,R,S, rdm );

    else if (M == Q && N == J && K == S && L == R)
      contribution += 1.0*getRDM(I,M,N,P,P,Q,R,S, rdm );

    else if (M == Q && N == J && Q == S && L == R)
      contribution += 1.0*getRDM(I,M,N,P,K,P,R,S, rdm );

    else if (M == Q && N == J && Q == R && K == S)
      contribution += 1.0*getRDM(I,M,N,P,L,P,R,S, rdm );

    else if (M == Q && N == J && Q == I && L == R)
      contribution += 1.0*getRDM(M,N,P,S,K,P,R,S, rdm );

    else if (M == Q && N == J && Q == I && K == S)
      contribution += 1.0*getRDM(M,N,P,R,L,P,R,S, rdm );

    else if (M == Q && N == J && P == S && L == R)
      contribution += -1.0*getRDM(I,M,N,P,K,Q,R,S, rdm );

    else if (M == Q && N == J && P == S && Q == R)
      contribution += 1.0*getRDM(I,M,N,P,K,L,R,S, rdm );

    else if (M == Q && N == J && P == S && Q == I)
      contribution += 1.0*getRDM(M,N,P,R,K,L,R,S, rdm );

    else if (M == Q && N == J && P == R && K == S)
      contribution += -1.0*getRDM(I,M,N,P,L,Q,R,S, rdm );

    else if (M == Q && N == J && P == R && Q == S)
      contribution += -1.0*getRDM(I,M,N,P,K,L,R,S, rdm );

    else if (M == Q && N == J && P == R && Q == I)
      contribution += -1.0*getRDM(M,N,P,S,K,L,R,S, rdm );

    else if (M == Q && N == I && K == S && L == R)
      contribution += -1.0*getRDM(J,M,N,P,P,Q,R,S, rdm );

    else if (M == Q && N == I && Q == S && L == R)
      contribution += -1.0*getRDM(J,M,N,P,K,P,R,S, rdm );

    else if (M == Q && N == I && Q == R && K == S)
      contribution += -1.0*getRDM(J,M,N,P,L,P,R,S, rdm );

    else if (M == Q && N == I && P == S && L == R)
      contribution += 1.0*getRDM(J,M,N,P,K,Q,R,S, rdm );

    else if (M == Q && N == I && P == S && Q == R)
      contribution += -1.0*getRDM(J,M,N,P,K,L,R,S, rdm );

    else if (M == Q && N == I && P == R && K == S)
      contribution += 1.0*getRDM(J,M,N,P,L,Q,R,S, rdm );

    else if (M == Q && N == I && P == R && Q == S)
      contribution += 1.0*getRDM(J,M,N,P,K,L,R,S, rdm );

    else if (M == Q && N == I && P == J && L == R)
      contribution += 1.0*getRDM(M,N,P,S,K,Q,R,S, rdm );

    else if (M == Q && N == I && P == J && K == S)
      contribution += 1.0*getRDM(M,N,P,R,L,Q,R,S, rdm );

    else if (M == Q && N == I && P == J && Q == S)
      contribution += 1.0*getRDM(M,N,P,R,K,L,R,S, rdm );

    else if (M == Q && N == I && P == J && Q == R)
      contribution += -1.0*getRDM(M,N,P,S,K,L,R,S, rdm );

    else if (M == Q && N == P && K == S && L == R)
      contribution += -1.0*getRDM(I,J,M,N,P,Q,R,S, rdm );

    else if (M == Q && N == P && Q == S && L == R)
      contribution += -1.0*getRDM(I,J,M,N,K,P,R,S, rdm );

    else if (M == Q && N == P && Q == R && K == S)
      contribution += -1.0*getRDM(I,J,M,N,L,P,R,S, rdm );

    else if (M == Q && N == P && Q == I && L == R)
      contribution += -1.0*getRDM(J,M,N,S,K,P,R,S, rdm );

    else if (M == Q && N == P && Q == I && K == S)
      contribution += -1.0*getRDM(J,M,N,R,L,P,R,S, rdm );

    else if (M == Q && N == P && P == S && L == R)
      contribution += 1.0*getRDM(I,J,M,N,K,Q,R,S, rdm );

    else if (M == Q && N == P && P == S && Q == R)
      contribution += -1.0*getRDM(I,J,M,N,K,L,R,S, rdm );

    else if (M == Q && N == P && P == S && Q == I)
      contribution += -1.0*getRDM(J,M,N,R,K,L,R,S, rdm );

    else if (M == Q && N == P && P == R && K == S)
      contribution += 1.0*getRDM(I,J,M,N,L,Q,R,S, rdm );

    else if (M == Q && N == P && P == R && Q == S)
      contribution += 1.0*getRDM(I,J,M,N,K,L,R,S, rdm );

    else if (M == Q && N == P && P == R && Q == I)
      contribution += 1.0*getRDM(J,M,N,S,K,L,R,S, rdm );

    else if (M == Q && N == P && P == J && L == R)
      contribution += -1.0*getRDM(I,M,N,S,K,Q,R,S, rdm );

    else if (M == Q && N == P && P == J && K == S)
      contribution += -1.0*getRDM(I,M,N,R,L,Q,R,S, rdm );

    else if (M == Q && N == P && P == J && Q == S)
      contribution += -1.0*getRDM(I,M,N,R,K,L,R,S, rdm );

    else if (M == Q && N == P && P == J && Q == R)
      contribution += 1.0*getRDM(I,M,N,S,K,L,R,S, rdm );

    else if (M == Q && N == P && P == J && Q == I)
      contribution += -1.0*getRDM(M,N,R,S,K,L,R,S, rdm );

    else if (M == P && Q == I && K == S && L == R)
      contribution += 1.0*getRDM(J,M,N,Q,N,P,R,S, rdm );

    else if (M == P && P == S && Q == I && L == R)
      contribution += 1.0*getRDM(J,M,N,Q,K,N,R,S, rdm );

    else if (M == P && P == R && Q == I && K == S)
      contribution += 1.0*getRDM(J,M,N,Q,L,N,R,S, rdm );

    else if (M == P && P == J && K == S && L == R)
      contribution += 1.0*getRDM(I,M,N,Q,N,Q,R,S, rdm );

    else if (M == P && P == J && Q == S && L == R)
      contribution += 1.0*getRDM(I,M,N,Q,K,N,R,S, rdm );

    else if (M == P && P == J && Q == R && K == S)
      contribution += 1.0*getRDM(I,M,N,Q,L,N,R,S, rdm );

    else if (M == P && P == J && Q == I && L == R)
      contribution += 1.0*getRDM(M,N,Q,S,K,N,R,S, rdm );

    else if (M == P && P == J && Q == I && K == S)
      contribution += 1.0*getRDM(M,N,Q,R,L,N,R,S, rdm );

    else if (M == P && N == S && Q == I && L == R)
      contribution += -1.0*getRDM(J,M,N,Q,K,P,R,S, rdm );

    else if (M == P && N == S && P == R && Q == I)
      contribution += 1.0*getRDM(J,M,N,Q,K,L,R,S, rdm );

    else if (M == P && N == S && P == J && L == R)
      contribution += -1.0*getRDM(I,M,N,Q,K,Q,R,S, rdm );

    else if (M == P && N == S && P == J && Q == R)
      contribution += 1.0*getRDM(I,M,N,Q,K,L,R,S, rdm );

    else if (M == P && N == S && P == J && Q == I)
      contribution += 1.0*getRDM(M,N,Q,R,K,L,R,S, rdm );

    else if (M == P && N == R && Q == I && K == S)
      contribution += -1.0*getRDM(J,M,N,Q,L,P,R,S, rdm );

    else if (M == P && N == R && P == S && Q == I)
      contribution += -1.0*getRDM(J,M,N,Q,K,L,R,S, rdm );

    else if (M == P && N == R && P == J && K == S)
      contribution += -1.0*getRDM(I,M,N,Q,L,Q,R,S, rdm );

    else if (M == P && N == R && P == J && Q == S)
      contribution += -1.0*getRDM(I,M,N,Q,K,L,R,S, rdm );

    else if (M == P && N == R && P == J && Q == I)
      contribution += -1.0*getRDM(M,N,Q,S,K,L,R,S, rdm );

    else if (M == P && N == J && K == S && L == R)
      contribution += -1.0*getRDM(I,M,N,Q,P,Q,R,S, rdm );

    else if (M == P && N == J && Q == S && L == R)
      contribution += -1.0*getRDM(I,M,N,Q,K,P,R,S, rdm );

    else if (M == P && N == J && Q == R && K == S)
      contribution += -1.0*getRDM(I,M,N,Q,L,P,R,S, rdm );

    else if (M == P && N == J && Q == I && L == R)
      contribution += -1.0*getRDM(M,N,Q,S,K,P,R,S, rdm );

    else if (M == P && N == J && Q == I && K == S)
      contribution += -1.0*getRDM(M,N,Q,R,L,P,R,S, rdm );

    else if (M == P && N == J && P == S && L == R)
      contribution += 1.0*getRDM(I,M,N,Q,K,Q,R,S, rdm );

    else if (M == P && N == J && P == S && Q == R)
      contribution += -1.0*getRDM(I,M,N,Q,K,L,R,S, rdm );

    else if (M == P && N == J && P == S && Q == I)
      contribution += -1.0*getRDM(M,N,Q,R,K,L,R,S, rdm );

    else if (M == P && N == J && P == R && K == S)
      contribution += 1.0*getRDM(I,M,N,Q,L,Q,R,S, rdm );

    else if (M == P && N == J && P == R && Q == S)
      contribution += 1.0*getRDM(I,M,N,Q,K,L,R,S, rdm );

    else if (M == P && N == J && P == R && Q == I)
      contribution += 1.0*getRDM(M,N,Q,S,K,L,R,S, rdm );

    else if (M == P && N == I && K == S && L == R)
      contribution += 1.0*getRDM(J,M,N,Q,P,Q,R,S, rdm );

    else if (M == P && N == I && Q == S && L == R)
      contribution += 1.0*getRDM(J,M,N,Q,K,P,R,S, rdm );

    else if (M == P && N == I && Q == R && K == S)
      contribution += 1.0*getRDM(J,M,N,Q,L,P,R,S, rdm );

    else if (M == P && N == I && P == S && L == R)
      contribution += -1.0*getRDM(J,M,N,Q,K,Q,R,S, rdm );

    else if (M == P && N == I && P == S && Q == R)
      contribution += 1.0*getRDM(J,M,N,Q,K,L,R,S, rdm );

    else if (M == P && N == I && P == R && K == S)
      contribution += -1.0*getRDM(J,M,N,Q,L,Q,R,S, rdm );

    else if (M == P && N == I && P == R && Q == S)
      contribution += -1.0*getRDM(J,M,N,Q,K,L,R,S, rdm );

    else if (M == P && N == I && P == J && L == R)
      contribution += -1.0*getRDM(M,N,Q,S,K,Q,R,S, rdm );

    else if (M == P && N == I && P == J && K == S)
      contribution += -1.0*getRDM(M,N,Q,R,L,Q,R,S, rdm );

    else if (M == P && N == I && P == J && Q == S)
      contribution += -1.0*getRDM(M,N,Q,R,K,L,R,S, rdm );

    else if (M == P && N == I && P == J && Q == R)
      contribution += 1.0*getRDM(M,N,Q,S,K,L,R,S, rdm );

    else if (M == P && N == Q && K == S && L == R)
      contribution += 1.0*getRDM(I,J,M,N,P,Q,R,S, rdm );

    else if (M == P && N == Q && Q == S && L == R)
      contribution += 1.0*getRDM(I,J,M,N,K,P,R,S, rdm );

    else if (M == P && N == Q && Q == R && K == S)
      contribution += 1.0*getRDM(I,J,M,N,L,P,R,S, rdm );

    else if (M == P && N == Q && Q == I && L == R)
      contribution += 1.0*getRDM(J,M,N,S,K,P,R,S, rdm );

    else if (M == P && N == Q && Q == I && K == S)
      contribution += 1.0*getRDM(J,M,N,R,L,P,R,S, rdm );

    else if (M == P && N == Q && P == S && L == R)
      contribution += -1.0*getRDM(I,J,M,N,K,Q,R,S, rdm );

    else if (M == P && N == Q && P == S && Q == R)
      contribution += 1.0*getRDM(I,J,M,N,K,L,R,S, rdm );

    else if (M == P && N == Q && P == S && Q == I)
      contribution += 1.0*getRDM(J,M,N,R,K,L,R,S, rdm );

    else if (M == P && N == Q && P == R && K == S)
      contribution += -1.0*getRDM(I,J,M,N,L,Q,R,S, rdm );

    else if (M == P && N == Q && P == R && Q == S)
      contribution += -1.0*getRDM(I,J,M,N,K,L,R,S, rdm );

    else if (M == P && N == Q && P == R && Q == I)
      contribution += -1.0*getRDM(J,M,N,S,K,L,R,S, rdm );

    else if (M == P && N == Q && P == J && L == R)
      contribution += 1.0*getRDM(I,M,N,S,K,Q,R,S, rdm );

    else if (M == P && N == Q && P == J && K == S)
      contribution += 1.0*getRDM(I,M,N,R,L,Q,R,S, rdm );

    else if (M == P && N == Q && P == J && Q == S)
      contribution += 1.0*getRDM(I,M,N,R,K,L,R,S, rdm );

    else if (M == P && N == Q && P == J && Q == R)
      contribution += -1.0*getRDM(I,M,N,S,K,L,R,S, rdm );

    else if (M == P && N == Q && P == J && Q == I)
      contribution += 1.0*getRDM(M,N,R,S,K,L,R,S, rdm );

    else if (Q == I && K == S && L == R)
      contribution += 1.0*getRDM(J,M,N,P,Q,M,N,P,R,S, rdm );

    else if (P == S && Q == I && L == R)
      contribution += -1.0*getRDM(J,M,N,P,Q,K,M,N,R,S, rdm );

    else if (P == R && Q == I && K == S)
      contribution += -1.0*getRDM(J,M,N,P,Q,L,M,N,R,S, rdm );

    else if (P == J && K == S && L == R)
      contribution += 1.0*getRDM(I,M,N,P,Q,M,N,Q,R,S, rdm );

    else if (P == J && Q == S && L == R)
      contribution += -1.0*getRDM(I,M,N,P,Q,K,M,N,R,S, rdm );

    else if (P == J && Q == R && K == S)
      contribution += -1.0*getRDM(I,M,N,P,Q,L,M,N,R,S, rdm );

    else if (P == J && Q == I && L == R)
      contribution += 1.0*getRDM(M,N,P,Q,S,K,M,N,R,S, rdm );

    else if (P == J && Q == I && K == S)
      contribution += 1.0*getRDM(M,N,P,Q,R,L,M,N,R,S, rdm );

    else if (N == S && Q == I && L == R)
      contribution += 1.0*getRDM(J,M,N,P,Q,K,M,P,R,S, rdm );

    else if (N == S && P == R && Q == I)
      contribution += 1.0*getRDM(J,M,N,P,Q,K,L,M,R,S, rdm );

    else if (N == S && P == J && L == R)
      contribution += 1.0*getRDM(I,M,N,P,Q,K,M,Q,R,S, rdm );

    else if (N == S && P == J && Q == R)
      contribution += 1.0*getRDM(I,M,N,P,Q,K,L,M,R,S, rdm );

    else if (N == S && P == J && Q == I)
      contribution += -1.0*getRDM(M,N,P,Q,R,K,L,M,R,S, rdm );

    else if (N == R && Q == I && K == S)
      contribution += 1.0*getRDM(J,M,N,P,Q,L,M,P,R,S, rdm );

    else if (N == R && P == S && Q == I)
      contribution += -1.0*getRDM(J,M,N,P,Q,K,L,M,R,S, rdm );

    else if (N == R && P == J && K == S)
      contribution += 1.0*getRDM(I,M,N,P,Q,L,M,Q,R,S, rdm );

    else if (N == R && P == J && Q == S)
      contribution += -1.0*getRDM(I,M,N,P,Q,K,L,M,R,S, rdm );

    else if (N == R && P == J && Q == I)
      contribution += 1.0*getRDM(M,N,P,Q,S,K,L,M,R,S, rdm );

    else if (N == J && K == S && L == R)
      contribution += -1.0*getRDM(I,M,N,P,Q,M,P,Q,R,S, rdm );

    else if (N == J && Q == S && L == R)
      contribution += 1.0*getRDM(I,M,N,P,Q,K,M,P,R,S, rdm );

    else if (N == J && Q == R && K == S)
      contribution += 1.0*getRDM(I,M,N,P,Q,L,M,P,R,S, rdm );

    else if (N == J && Q == I && L == R)
      contribution += -1.0*getRDM(M,N,P,Q,S,K,M,P,R,S, rdm );

    else if (N == J && Q == I && K == S)
      contribution += -1.0*getRDM(M,N,P,Q,R,L,M,P,R,S, rdm );

    else if (N == J && P == S && L == R)
      contribution += -1.0*getRDM(I,M,N,P,Q,K,M,Q,R,S, rdm );

    else if (N == J && P == S && Q == R)
      contribution += -1.0*getRDM(I,M,N,P,Q,K,L,M,R,S, rdm );

    else if (N == J && P == S && Q == I)
      contribution += 1.0*getRDM(M,N,P,Q,R,K,L,M,R,S, rdm );

    else if (N == J && P == R && K == S)
      contribution += -1.0*getRDM(I,M,N,P,Q,L,M,Q,R,S, rdm );

    else if (N == J && P == R && Q == S)
      contribution += 1.0*getRDM(I,M,N,P,Q,K,L,M,R,S, rdm );

    else if (N == J && P == R && Q == I)
      contribution += -1.0*getRDM(M,N,P,Q,S,K,L,M,R,S, rdm );

    else if (N == I && K == S && L == R)
      contribution += 1.0*getRDM(J,M,N,P,Q,M,P,Q,R,S, rdm );

    else if (N == I && Q == S && L == R)
      contribution += -1.0*getRDM(J,M,N,P,Q,K,M,P,R,S, rdm );

    else if (N == I && Q == R && K == S)
      contribution += -1.0*getRDM(J,M,N,P,Q,L,M,P,R,S, rdm );

    else if (N == I && P == S && L == R)
      contribution += 1.0*getRDM(J,M,N,P,Q,K,M,Q,R,S, rdm );

    else if (N == I && P == S && Q == R)
      contribution += 1.0*getRDM(J,M,N,P,Q,K,L,M,R,S, rdm );

    else if (N == I && P == R && K == S)
      contribution += 1.0*getRDM(J,M,N,P,Q,L,M,Q,R,S, rdm );

    else if (N == I && P == R && Q == S)
      contribution += -1.0*getRDM(J,M,N,P,Q,K,L,M,R,S, rdm );

    else if (N == I && P == J && L == R)
      contribution += -1.0*getRDM(M,N,P,Q,S,K,M,Q,R,S, rdm );

    else if (N == I && P == J && K == S)
      contribution += -1.0*getRDM(M,N,P,Q,R,L,M,Q,R,S, rdm );

    else if (N == I && P == J && Q == S)
      contribution += 1.0*getRDM(M,N,P,Q,R,K,L,M,R,S, rdm );

    else if (N == I && P == J && Q == R)
      contribution += -1.0*getRDM(M,N,P,Q,S,K,L,M,R,S, rdm );

    else if (N == Q && K == S && L == R)
      contribution += -1.0*getRDM(I,J,M,N,P,M,P,Q,R,S, rdm );

    else if (N == Q && Q == S && L == R)
      contribution += 1.0*getRDM(I,J,M,N,P,K,M,P,R,S, rdm );

    else if (N == Q && Q == R && K == S)
      contribution += 1.0*getRDM(I,J,M,N,P,L,M,P,R,S, rdm );

    else if (N == Q && Q == I && L == R)
      contribution += -1.0*getRDM(J,M,N,P,S,K,M,P,R,S, rdm );

    else if (N == Q && Q == I && K == S)
      contribution += -1.0*getRDM(J,M,N,P,R,L,M,P,R,S, rdm );

    else if (N == Q && P == S && L == R)
      contribution += -1.0*getRDM(I,J,M,N,P,K,M,Q,R,S, rdm );

    else if (N == Q && P == S && Q == R)
      contribution += -1.0*getRDM(I,J,M,N,P,K,L,M,R,S, rdm );

    else if (N == Q && P == S && Q == I)
      contribution += 1.0*getRDM(J,M,N,P,R,K,L,M,R,S, rdm );

    else if (N == Q && P == R && K == S)
      contribution += -1.0*getRDM(I,J,M,N,P,L,M,Q,R,S, rdm );

    else if (N == Q && P == R && Q == S)
      contribution += 1.0*getRDM(I,J,M,N,P,K,L,M,R,S, rdm );

    else if (N == Q && P == R && Q == I)
      contribution += -1.0*getRDM(J,M,N,P,S,K,L,M,R,S, rdm );

    else if (N == Q && P == J && L == R)
      contribution += -1.0*getRDM(I,M,N,P,S,K,M,Q,R,S, rdm );

    else if (N == Q && P == J && K == S)
      contribution += -1.0*getRDM(I,M,N,P,R,L,M,Q,R,S, rdm );

    else if (N == Q && P == J && Q == S)
      contribution += 1.0*getRDM(I,M,N,P,R,K,L,M,R,S, rdm );

    else if (N == Q && P == J && Q == R)
      contribution += -1.0*getRDM(I,M,N,P,S,K,L,M,R,S, rdm );

    else if (N == Q && P == J && Q == I)
      contribution += -1.0*getRDM(M,N,P,R,S,K,L,M,R,S, rdm );

    else if (N == P && K == S && L == R)
      contribution += 1.0*getRDM(I,J,M,N,Q,M,P,Q,R,S, rdm );

    else if (N == P && Q == S && L == R)
      contribution += -1.0*getRDM(I,J,M,N,Q,K,M,P,R,S, rdm );

    else if (N == P && Q == R && K == S)
      contribution += -1.0*getRDM(I,J,M,N,Q,L,M,P,R,S, rdm );

    else if (N == P && Q == I && L == R)
      contribution += 1.0*getRDM(J,M,N,Q,S,K,M,P,R,S, rdm );

    else if (N == P && Q == I && K == S)
      contribution += 1.0*getRDM(J,M,N,Q,R,L,M,P,R,S, rdm );

    else if (N == P && P == S && L == R)
      contribution += 1.0*getRDM(I,J,M,N,Q,K,M,Q,R,S, rdm );

    else if (N == P && P == S && Q == R)
      contribution += 1.0*getRDM(I,J,M,N,Q,K,L,M,R,S, rdm );

    else if (N == P && P == S && Q == I)
      contribution += -1.0*getRDM(J,M,N,Q,R,K,L,M,R,S, rdm );

    else if (N == P && P == R && K == S)
      contribution += 1.0*getRDM(I,J,M,N,Q,L,M,Q,R,S, rdm );

    else if (N == P && P == R && Q == S)
      contribution += -1.0*getRDM(I,J,M,N,Q,K,L,M,R,S, rdm );

    else if (N == P && P == R && Q == I)
      contribution += 1.0*getRDM(J,M,N,Q,S,K,L,M,R,S, rdm );

    else if (N == P && P == J && L == R)
      contribution += 1.0*getRDM(I,M,N,Q,S,K,M,Q,R,S, rdm );

    else if (N == P && P == J && K == S)
      contribution += 1.0*getRDM(I,M,N,Q,R,L,M,Q,R,S, rdm );

    else if (N == P && P == J && Q == S)
      contribution += -1.0*getRDM(I,M,N,Q,R,K,L,M,R,S, rdm );

    else if (N == P && P == J && Q == R)
      contribution += 1.0*getRDM(I,M,N,Q,S,K,L,M,R,S, rdm );

    else if (N == P && P == J && Q == I)
      contribution += 1.0*getRDM(M,N,Q,R,S,K,L,M,R,S, rdm );

    else if (M == S && Q == I && L == R)
      contribution += -1.0*getRDM(J,M,N,P,Q,K,N,P,R,S, rdm );

    else if (M == S && P == R && Q == I)
      contribution += -1.0*getRDM(J,M,N,P,Q,K,L,N,R,S, rdm );

    else if (M == S && P == J && L == R)
      contribution += -1.0*getRDM(I,M,N,P,Q,K,N,Q,R,S, rdm );

    else if (M == S && P == J && Q == R)
      contribution += -1.0*getRDM(I,M,N,P,Q,K,L,N,R,S, rdm );

    else if (M == S && P == J && Q == I)
      contribution += 1.0*getRDM(M,N,P,Q,R,K,L,N,R,S, rdm );

    else if (M == S && N == R && Q == I)
      contribution += 1.0*getRDM(J,M,N,P,Q,K,L,P,R,S, rdm );

    else if (M == S && N == R && P == J)
      contribution += 1.0*getRDM(I,M,N,P,Q,K,L,Q,R,S, rdm );

    else if (M == S && N == J && L == R)
      contribution += 1.0*getRDM(I,M,N,P,Q,K,P,Q,R,S, rdm );

    else if (M == S && N == J && Q == R)
      contribution += 1.0*getRDM(I,M,N,P,Q,K,L,P,R,S, rdm );

    else if (M == S && N == J && Q == I)
      contribution += -1.0*getRDM(M,N,P,Q,R,K,L,P,R,S, rdm );

    else if (M == S && N == J && P == R)
      contribution += -1.0*getRDM(I,M,N,P,Q,K,L,Q,R,S, rdm );

    else if (M == S && N == I && L == R)
      contribution += -1.0*getRDM(J,M,N,P,Q,K,P,Q,R,S, rdm );

    else if (M == S && N == I && Q == R)
      contribution += -1.0*getRDM(J,M,N,P,Q,K,L,P,R,S, rdm );

    else if (M == S && N == I && P == R)
      contribution += 1.0*getRDM(J,M,N,P,Q,K,L,Q,R,S, rdm );

    else if (M == S && N == I && P == J)
      contribution += -1.0*getRDM(M,N,P,Q,R,K,L,Q,R,S, rdm );

    else if (M == S && N == Q && L == R)
      contribution += 1.0*getRDM(I,J,M,N,P,K,P,Q,R,S, rdm );

    else if (M == S && N == Q && Q == R)
      contribution += 1.0*getRDM(I,J,M,N,P,K,L,P,R,S, rdm );

    else if (M == S && N == Q && Q == I)
      contribution += -1.0*getRDM(J,M,N,P,R,K,L,P,R,S, rdm );

    else if (M == S && N == Q && P == R)
      contribution += -1.0*getRDM(I,J,M,N,P,K,L,Q,R,S, rdm );

    else if (M == S && N == Q && P == J)
      contribution += -1.0*getRDM(I,M,N,P,R,K,L,Q,R,S, rdm );

    else if (M == S && N == P && L == R)
      contribution += -1.0*getRDM(I,J,M,N,Q,K,P,Q,R,S, rdm );

    else if (M == S && N == P && Q == R)
      contribution += -1.0*getRDM(I,J,M,N,Q,K,L,P,R,S, rdm );

    else if (M == S && N == P && Q == I)
      contribution += 1.0*getRDM(J,M,N,Q,R,K,L,P,R,S, rdm );

    else if (M == S && N == P && P == R)
      contribution += 1.0*getRDM(I,J,M,N,Q,K,L,Q,R,S, rdm );

    else if (M == S && N == P && P == J)
      contribution += 1.0*getRDM(I,M,N,Q,R,K,L,Q,R,S, rdm );

    else if (M == R && Q == I && K == S)
      contribution += -1.0*getRDM(J,M,N,P,Q,L,N,P,R,S, rdm );

    else if (M == R && P == S && Q == I)
      contribution += 1.0*getRDM(J,M,N,P,Q,K,L,N,R,S, rdm );

    else if (M == R && P == J && K == S)
      contribution += -1.0*getRDM(I,M,N,P,Q,L,N,Q,R,S, rdm );

    else if (M == R && P == J && Q == S)
      contribution += 1.0*getRDM(I,M,N,P,Q,K,L,N,R,S, rdm );

    else if (M == R && P == J && Q == I)
      contribution += -1.0*getRDM(M,N,P,Q,S,K,L,N,R,S, rdm );

    else if (M == R && N == S && Q == I)
      contribution += -1.0*getRDM(J,M,N,P,Q,K,L,P,R,S, rdm );

    else if (M == R && N == S && P == J)
      contribution += -1.0*getRDM(I,M,N,P,Q,K,L,Q,R,S, rdm );

    else if (M == R && N == J && K == S)
      contribution += 1.0*getRDM(I,M,N,P,Q,L,P,Q,R,S, rdm );

    else if (M == R && N == J && Q == S)
      contribution += -1.0*getRDM(I,M,N,P,Q,K,L,P,R,S, rdm );

    else if (M == R && N == J && Q == I)
      contribution += 1.0*getRDM(M,N,P,Q,S,K,L,P,R,S, rdm );

    else if (M == R && N == J && P == S)
      contribution += 1.0*getRDM(I,M,N,P,Q,K,L,Q,R,S, rdm );

    else if (M == R && N == I && K == S)
      contribution += -1.0*getRDM(J,M,N,P,Q,L,P,Q,R,S, rdm );

    else if (M == R && N == I && Q == S)
      contribution += 1.0*getRDM(J,M,N,P,Q,K,L,P,R,S, rdm );

    else if (M == R && N == I && P == S)
      contribution += -1.0*getRDM(J,M,N,P,Q,K,L,Q,R,S, rdm );

    else if (M == R && N == I && P == J)
      contribution += 1.0*getRDM(M,N,P,Q,S,K,L,Q,R,S, rdm );

    else if (M == R && N == Q && K == S)
      contribution += 1.0*getRDM(I,J,M,N,P,L,P,Q,R,S, rdm );

    else if (M == R && N == Q && Q == S)
      contribution += -1.0*getRDM(I,J,M,N,P,K,L,P,R,S, rdm );

    else if (M == R && N == Q && Q == I)
      contribution += 1.0*getRDM(J,M,N,P,S,K,L,P,R,S, rdm );

    else if (M == R && N == Q && P == S)
      contribution += 1.0*getRDM(I,J,M,N,P,K,L,Q,R,S, rdm );

    else if (M == R && N == Q && P == J)
      contribution += 1.0*getRDM(I,M,N,P,S,K,L,Q,R,S, rdm );

    else if (M == R && N == P && K == S)
      contribution += -1.0*getRDM(I,J,M,N,Q,L,P,Q,R,S, rdm );

    else if (M == R && N == P && Q == S)
      contribution += 1.0*getRDM(I,J,M,N,Q,K,L,P,R,S, rdm );

    else if (M == R && N == P && Q == I)
      contribution += -1.0*getRDM(J,M,N,Q,S,K,L,P,R,S, rdm );

    else if (M == R && N == P && P == S)
      contribution += -1.0*getRDM(I,J,M,N,Q,K,L,Q,R,S, rdm );

    else if (M == R && N == P && P == J)
      contribution += -1.0*getRDM(I,M,N,Q,S,K,L,Q,R,S, rdm );

    else if (M == J && K == S && L == R)
      contribution += 1.0*getRDM(I,M,N,P,Q,N,P,Q,R,S, rdm );

    else if (M == J && Q == S && L == R)
      contribution += -1.0*getRDM(I,M,N,P,Q,K,N,P,R,S, rdm );

    else if (M == J && Q == R && K == S)
      contribution += -1.0*getRDM(I,M,N,P,Q,L,N,P,R,S, rdm );

    else if (M == J && Q == I && L == R)
      contribution += 1.0*getRDM(M,N,P,Q,S,K,N,P,R,S, rdm );

    else if (M == J && Q == I && K == S)
      contribution += 1.0*getRDM(M,N,P,Q,R,L,N,P,R,S, rdm );

    else if (M == J && P == S && L == R)
      contribution += 1.0*getRDM(I,M,N,P,Q,K,N,Q,R,S, rdm );

    else if (M == J && P == S && Q == R)
      contribution += 1.0*getRDM(I,M,N,P,Q,K,L,N,R,S, rdm );

    else if (M == J && P == S && Q == I)
      contribution += -1.0*getRDM(M,N,P,Q,R,K,L,N,R,S, rdm );

    else if (M == J && P == R && K == S)
      contribution += 1.0*getRDM(I,M,N,P,Q,L,N,Q,R,S, rdm );

    else if (M == J && P == R && Q == S)
      contribution += -1.0*getRDM(I,M,N,P,Q,K,L,N,R,S, rdm );

    else if (M == J && P == R && Q == I)
      contribution += 1.0*getRDM(M,N,P,Q,S,K,L,N,R,S, rdm );

    else if (M == J && N == S && L == R)
      contribution += -1.0*getRDM(I,M,N,P,Q,K,P,Q,R,S, rdm );

    else if (M == J && N == S && Q == R)
      contribution += -1.0*getRDM(I,M,N,P,Q,K,L,P,R,S, rdm );

    else if (M == J && N == S && Q == I)
      contribution += 1.0*getRDM(M,N,P,Q,R,K,L,P,R,S, rdm );

    else if (M == J && N == S && P == R)
      contribution += 1.0*getRDM(I,M,N,P,Q,K,L,Q,R,S, rdm );

    else if (M == J && N == R && K == S)
      contribution += -1.0*getRDM(I,M,N,P,Q,L,P,Q,R,S, rdm );

    else if (M == J && N == R && Q == S)
      contribution += 1.0*getRDM(I,M,N,P,Q,K,L,P,R,S, rdm );

    else if (M == J && N == R && Q == I)
      contribution += -1.0*getRDM(M,N,P,Q,S,K,L,P,R,S, rdm );

    else if (M == J && N == R && P == S)
      contribution += -1.0*getRDM(I,M,N,P,Q,K,L,Q,R,S, rdm );

    else if (M == J && N == I && L == R)
      contribution += 1.0*getRDM(M,N,P,Q,S,K,P,Q,R,S, rdm );

    else if (M == J && N == I && K == S)
      contribution += 1.0*getRDM(M,N,P,Q,R,L,P,Q,R,S, rdm );

    else if (M == J && N == I && Q == S)
      contribution += -1.0*getRDM(M,N,P,Q,R,K,L,P,R,S, rdm );

    else if (M == J && N == I && Q == R)
      contribution += 1.0*getRDM(M,N,P,Q,S,K,L,P,R,S, rdm );

    else if (M == J && N == I && P == S)
      contribution += 1.0*getRDM(M,N,P,Q,R,K,L,Q,R,S, rdm );

    else if (M == J && N == I && P == R)
      contribution += -1.0*getRDM(M,N,P,Q,S,K,L,Q,R,S, rdm );

    else if (M == J && N == Q && L == R)
      contribution += 1.0*getRDM(I,M,N,P,S,K,P,Q,R,S, rdm );

    else if (M == J && N == Q && K == S)
      contribution += 1.0*getRDM(I,M,N,P,R,L,P,Q,R,S, rdm );

    else if (M == J && N == Q && Q == S)
      contribution += -1.0*getRDM(I,M,N,P,R,K,L,P,R,S, rdm );

    else if (M == J && N == Q && Q == R)
      contribution += 1.0*getRDM(I,M,N,P,S,K,L,P,R,S, rdm );

    else if (M == J && N == Q && Q == I)
      contribution += 1.0*getRDM(M,N,P,R,S,K,L,P,R,S, rdm );

    else if (M == J && N == Q && P == S)
      contribution += 1.0*getRDM(I,M,N,P,R,K,L,Q,R,S, rdm );

    else if (M == J && N == Q && P == R)
      contribution += -1.0*getRDM(I,M,N,P,S,K,L,Q,R,S, rdm );

    else if (M == J && N == P && L == R)
      contribution += -1.0*getRDM(I,M,N,Q,S,K,P,Q,R,S, rdm );

    else if (M == J && N == P && K == S)
      contribution += -1.0*getRDM(I,M,N,Q,R,L,P,Q,R,S, rdm );

    else if (M == J && N == P && Q == S)
      contribution += 1.0*getRDM(I,M,N,Q,R,K,L,P,R,S, rdm );

    else if (M == J && N == P && Q == R)
      contribution += -1.0*getRDM(I,M,N,Q,S,K,L,P,R,S, rdm );

    else if (M == J && N == P && Q == I)
      contribution += -1.0*getRDM(M,N,Q,R,S,K,L,P,R,S, rdm );

    else if (M == J && N == P && P == S)
      contribution += -1.0*getRDM(I,M,N,Q,R,K,L,Q,R,S, rdm );

    else if (M == J && N == P && P == R)
      contribution += 1.0*getRDM(I,M,N,Q,S,K,L,Q,R,S, rdm );

    else if (M == I && K == S && L == R)
      contribution += -1.0*getRDM(J,M,N,P,Q,N,P,Q,R,S, rdm );

    else if (M == I && Q == S && L == R)
      contribution += 1.0*getRDM(J,M,N,P,Q,K,N,P,R,S, rdm );

    else if (M == I && Q == R && K == S)
      contribution += 1.0*getRDM(J,M,N,P,Q,L,N,P,R,S, rdm );

    else if (M == I && P == S && L == R)
      contribution += -1.0*getRDM(J,M,N,P,Q,K,N,Q,R,S, rdm );

    else if (M == I && P == S && Q == R)
      contribution += -1.0*getRDM(J,M,N,P,Q,K,L,N,R,S, rdm );

    else if (M == I && P == R && K == S)
      contribution += -1.0*getRDM(J,M,N,P,Q,L,N,Q,R,S, rdm );

    else if (M == I && P == R && Q == S)
      contribution += 1.0*getRDM(J,M,N,P,Q,K,L,N,R,S, rdm );

    else if (M == I && P == J && L == R)
      contribution += 1.0*getRDM(M,N,P,Q,S,K,N,Q,R,S, rdm );

    else if (M == I && P == J && K == S)
      contribution += 1.0*getRDM(M,N,P,Q,R,L,N,Q,R,S, rdm );

    else if (M == I && P == J && Q == S)
      contribution += -1.0*getRDM(M,N,P,Q,R,K,L,N,R,S, rdm );

    else if (M == I && P == J && Q == R)
      contribution += 1.0*getRDM(M,N,P,Q,S,K,L,N,R,S, rdm );

    else if (M == I && N == S && L == R)
      contribution += 1.0*getRDM(J,M,N,P,Q,K,P,Q,R,S, rdm );

    else if (M == I && N == S && Q == R)
      contribution += 1.0*getRDM(J,M,N,P,Q,K,L,P,R,S, rdm );

    else if (M == I && N == S && P == R)
      contribution += -1.0*getRDM(J,M,N,P,Q,K,L,Q,R,S, rdm );

    else if (M == I && N == S && P == J)
      contribution += 1.0*getRDM(M,N,P,Q,R,K,L,Q,R,S, rdm );

    else if (M == I && N == R && K == S)
      contribution += 1.0*getRDM(J,M,N,P,Q,L,P,Q,R,S, rdm );

    else if (M == I && N == R && Q == S)
      contribution += -1.0*getRDM(J,M,N,P,Q,K,L,P,R,S, rdm );

    else if (M == I && N == R && P == S)
      contribution += 1.0*getRDM(J,M,N,P,Q,K,L,Q,R,S, rdm );

    else if (M == I && N == R && P == J)
      contribution += -1.0*getRDM(M,N,P,Q,S,K,L,Q,R,S, rdm );

    else if (M == I && N == J && L == R)
      contribution += -1.0*getRDM(M,N,P,Q,S,K,P,Q,R,S, rdm );

    else if (M == I && N == J && K == S)
      contribution += -1.0*getRDM(M,N,P,Q,R,L,P,Q,R,S, rdm );

    else if (M == I && N == J && Q == S)
      contribution += 1.0*getRDM(M,N,P,Q,R,K,L,P,R,S, rdm );

    else if (M == I && N == J && Q == R)
      contribution += -1.0*getRDM(M,N,P,Q,S,K,L,P,R,S, rdm );

    else if (M == I && N == J && P == S)
      contribution += -1.0*getRDM(M,N,P,Q,R,K,L,Q,R,S, rdm );

    else if (M == I && N == J && P == R)
      contribution += 1.0*getRDM(M,N,P,Q,S,K,L,Q,R,S, rdm );

    else if (M == I && N == Q && L == R)
      contribution += -1.0*getRDM(J,M,N,P,S,K,P,Q,R,S, rdm );

    else if (M == I && N == Q && K == S)
      contribution += -1.0*getRDM(J,M,N,P,R,L,P,Q,R,S, rdm );

    else if (M == I && N == Q && Q == S)
      contribution += 1.0*getRDM(J,M,N,P,R,K,L,P,R,S, rdm );

    else if (M == I && N == Q && Q == R)
      contribution += -1.0*getRDM(J,M,N,P,S,K,L,P,R,S, rdm );

    else if (M == I && N == Q && P == S)
      contribution += -1.0*getRDM(J,M,N,P,R,K,L,Q,R,S, rdm );

    else if (M == I && N == Q && P == R)
      contribution += 1.0*getRDM(J,M,N,P,S,K,L,Q,R,S, rdm );

    else if (M == I && N == Q && P == J)
      contribution += 1.0*getRDM(M,N,P,R,S,K,L,Q,R,S, rdm );

    else if (M == I && N == P && L == R)
      contribution += 1.0*getRDM(J,M,N,Q,S,K,P,Q,R,S, rdm );

    else if (M == I && N == P && K == S)
      contribution += 1.0*getRDM(J,M,N,Q,R,L,P,Q,R,S, rdm );

    else if (M == I && N == P && Q == S)
      contribution += -1.0*getRDM(J,M,N,Q,R,K,L,P,R,S, rdm );

    else if (M == I && N == P && Q == R)
      contribution += 1.0*getRDM(J,M,N,Q,S,K,L,P,R,S, rdm );

    else if (M == I && N == P && P == S)
      contribution += 1.0*getRDM(J,M,N,Q,R,K,L,Q,R,S, rdm );

    else if (M == I && N == P && P == R)
      contribution += -1.0*getRDM(J,M,N,Q,S,K,L,Q,R,S, rdm );

    else if (M == I && N == P && P == J)
      contribution += -1.0*getRDM(M,N,Q,R,S,K,L,Q,R,S, rdm );

    else if (M == Q && K == S && L == R)
      contribution += 1.0*getRDM(I,J,M,N,P,N,P,Q,R,S, rdm );

    else if (M == Q && Q == S && L == R)
      contribution += -1.0*getRDM(I,J,M,N,P,K,N,P,R,S, rdm );

    else if (M == Q && Q == R && K == S)
      contribution += -1.0*getRDM(I,J,M,N,P,L,N,P,R,S, rdm );

    else if (M == Q && Q == I && L == R)
      contribution += 1.0*getRDM(J,M,N,P,S,K,N,P,R,S, rdm );

    else if (M == Q && Q == I && K == S)
      contribution += 1.0*getRDM(J,M,N,P,R,L,N,P,R,S, rdm );

    else if (M == Q && P == S && L == R)
      contribution += 1.0*getRDM(I,J,M,N,P,K,N,Q,R,S, rdm );

    else if (M == Q && P == S && Q == R)
      contribution += 1.0*getRDM(I,J,M,N,P,K,L,N,R,S, rdm );

    else if (M == Q && P == S && Q == I)
      contribution += -1.0*getRDM(J,M,N,P,R,K,L,N,R,S, rdm );

    else if (M == Q && P == R && K == S)
      contribution += 1.0*getRDM(I,J,M,N,P,L,N,Q,R,S, rdm );

    else if (M == Q && P == R && Q == S)
      contribution += -1.0*getRDM(I,J,M,N,P,K,L,N,R,S, rdm );

    else if (M == Q && P == R && Q == I)
      contribution += 1.0*getRDM(J,M,N,P,S,K,L,N,R,S, rdm );

    else if (M == Q && P == J && L == R)
      contribution += 1.0*getRDM(I,M,N,P,S,K,N,Q,R,S, rdm );

    else if (M == Q && P == J && K == S)
      contribution += 1.0*getRDM(I,M,N,P,R,L,N,Q,R,S, rdm );

    else if (M == Q && P == J && Q == S)
      contribution += -1.0*getRDM(I,M,N,P,R,K,L,N,R,S, rdm );

    else if (M == Q && P == J && Q == R)
      contribution += 1.0*getRDM(I,M,N,P,S,K,L,N,R,S, rdm );

    else if (M == Q && P == J && Q == I)
      contribution += 1.0*getRDM(M,N,P,R,S,K,L,N,R,S, rdm );

    else if (M == Q && N == S && L == R)
      contribution += -1.0*getRDM(I,J,M,N,P,K,P,Q,R,S, rdm );

    else if (M == Q && N == S && Q == R)
      contribution += -1.0*getRDM(I,J,M,N,P,K,L,P,R,S, rdm );

    else if (M == Q && N == S && Q == I)
      contribution += 1.0*getRDM(J,M,N,P,R,K,L,P,R,S, rdm );

    else if (M == Q && N == S && P == R)
      contribution += 1.0*getRDM(I,J,M,N,P,K,L,Q,R,S, rdm );

    else if (M == Q && N == S && P == J)
      contribution += 1.0*getRDM(I,M,N,P,R,K,L,Q,R,S, rdm );

    else if (M == Q && N == R && K == S)
      contribution += -1.0*getRDM(I,J,M,N,P,L,P,Q,R,S, rdm );

    else if (M == Q && N == R && Q == S)
      contribution += 1.0*getRDM(I,J,M,N,P,K,L,P,R,S, rdm );

    else if (M == Q && N == R && Q == I)
      contribution += -1.0*getRDM(J,M,N,P,S,K,L,P,R,S, rdm );

    else if (M == Q && N == R && P == S)
      contribution += -1.0*getRDM(I,J,M,N,P,K,L,Q,R,S, rdm );

    else if (M == Q && N == R && P == J)
      contribution += -1.0*getRDM(I,M,N,P,S,K,L,Q,R,S, rdm );

    else if (M == Q && N == J && L == R)
      contribution += -1.0*getRDM(I,M,N,P,S,K,P,Q,R,S, rdm );

    else if (M == Q && N == J && K == S)
      contribution += -1.0*getRDM(I,M,N,P,R,L,P,Q,R,S, rdm );

    else if (M == Q && N == J && Q == S)
      contribution += 1.0*getRDM(I,M,N,P,R,K,L,P,R,S, rdm );

    else if (M == Q && N == J && Q == R)
      contribution += -1.0*getRDM(I,M,N,P,S,K,L,P,R,S, rdm );

    else if (M == Q && N == J && Q == I)
      contribution += -1.0*getRDM(M,N,P,R,S,K,L,P,R,S, rdm );

    else if (M == Q && N == J && P == S)
      contribution += -1.0*getRDM(I,M,N,P,R,K,L,Q,R,S, rdm );

    else if (M == Q && N == J && P == R)
      contribution += 1.0*getRDM(I,M,N,P,S,K,L,Q,R,S, rdm );

    else if (M == Q && N == I && L == R)
      contribution += 1.0*getRDM(J,M,N,P,S,K,P,Q,R,S, rdm );

    else if (M == Q && N == I && K == S)
      contribution += 1.0*getRDM(J,M,N,P,R,L,P,Q,R,S, rdm );

    else if (M == Q && N == I && Q == S)
      contribution += -1.0*getRDM(J,M,N,P,R,K,L,P,R,S, rdm );

    else if (M == Q && N == I && Q == R)
      contribution += 1.0*getRDM(J,M,N,P,S,K,L,P,R,S, rdm );

    else if (M == Q && N == I && P == S)
      contribution += 1.0*getRDM(J,M,N,P,R,K,L,Q,R,S, rdm );

    else if (M == Q && N == I && P == R)
      contribution += -1.0*getRDM(J,M,N,P,S,K,L,Q,R,S, rdm );

    else if (M == Q && N == I && P == J)
      contribution += -1.0*getRDM(M,N,P,R,S,K,L,Q,R,S, rdm );

    else if (M == Q && N == P && L == R)
      contribution += 1.0*getRDM(I,J,M,N,S,K,P,Q,R,S, rdm );

    else if (M == Q && N == P && K == S)
      contribution += 1.0*getRDM(I,J,M,N,R,L,P,Q,R,S, rdm );

    else if (M == Q && N == P && Q == S)
      contribution += -1.0*getRDM(I,J,M,N,R,K,L,P,R,S, rdm );

    else if (M == Q && N == P && Q == R)
      contribution += 1.0*getRDM(I,J,M,N,S,K,L,P,R,S, rdm );

    else if (M == Q && N == P && Q == I)
      contribution += 1.0*getRDM(J,M,N,R,S,K,L,P,R,S, rdm );

    else if (M == Q && N == P && P == S)
      contribution += 1.0*getRDM(I,J,M,N,R,K,L,Q,R,S, rdm );

    else if (M == Q && N == P && P == R)
      contribution += -1.0*getRDM(I,J,M,N,S,K,L,Q,R,S, rdm );

    else if (M == Q && N == P && P == J)
      contribution += 1.0*getRDM(I,M,N,R,S,K,L,Q,R,S, rdm );

    else if (M == P && K == S && L == R)
      contribution += -1.0*getRDM(I,J,M,N,Q,N,P,Q,R,S, rdm );

    else if (M == P && Q == S && L == R)
      contribution += 1.0*getRDM(I,J,M,N,Q,K,N,P,R,S, rdm );

    else if (M == P && Q == R && K == S)
      contribution += 1.0*getRDM(I,J,M,N,Q,L,N,P,R,S, rdm );

    else if (M == P && Q == I && L == R)
      contribution += -1.0*getRDM(J,M,N,Q,S,K,N,P,R,S, rdm );

    else if (M == P && Q == I && K == S)
      contribution += -1.0*getRDM(J,M,N,Q,R,L,N,P,R,S, rdm );

    else if (M == P && P == S && L == R)
      contribution += -1.0*getRDM(I,J,M,N,Q,K,N,Q,R,S, rdm );

    else if (M == P && P == S && Q == R)
      contribution += -1.0*getRDM(I,J,M,N,Q,K,L,N,R,S, rdm );

    else if (M == P && P == S && Q == I)
      contribution += 1.0*getRDM(J,M,N,Q,R,K,L,N,R,S, rdm );

    else if (M == P && P == R && K == S)
      contribution += -1.0*getRDM(I,J,M,N,Q,L,N,Q,R,S, rdm );

    else if (M == P && P == R && Q == S)
      contribution += 1.0*getRDM(I,J,M,N,Q,K,L,N,R,S, rdm );

    else if (M == P && P == R && Q == I)
      contribution += -1.0*getRDM(J,M,N,Q,S,K,L,N,R,S, rdm );

    else if (M == P && P == J && L == R)
      contribution += -1.0*getRDM(I,M,N,Q,S,K,N,Q,R,S, rdm );

    else if (M == P && P == J && K == S)
      contribution += -1.0*getRDM(I,M,N,Q,R,L,N,Q,R,S, rdm );

    else if (M == P && P == J && Q == S)
      contribution += 1.0*getRDM(I,M,N,Q,R,K,L,N,R,S, rdm );

    else if (M == P && P == J && Q == R)
      contribution += -1.0*getRDM(I,M,N,Q,S,K,L,N,R,S, rdm );

    else if (M == P && P == J && Q == I)
      contribution += -1.0*getRDM(M,N,Q,R,S,K,L,N,R,S, rdm );

    else if (M == P && N == S && L == R)
      contribution += 1.0*getRDM(I,J,M,N,Q,K,P,Q,R,S, rdm );

    else if (M == P && N == S && Q == R)
      contribution += 1.0*getRDM(I,J,M,N,Q,K,L,P,R,S, rdm );

    else if (M == P && N == S && Q == I)
      contribution += -1.0*getRDM(J,M,N,Q,R,K,L,P,R,S, rdm );

    else if (M == P && N == S && P == R)
      contribution += -1.0*getRDM(I,J,M,N,Q,K,L,Q,R,S, rdm );

    else if (M == P && N == S && P == J)
      contribution += -1.0*getRDM(I,M,N,Q,R,K,L,Q,R,S, rdm );

    else if (M == P && N == R && K == S)
      contribution += 1.0*getRDM(I,J,M,N,Q,L,P,Q,R,S, rdm );

    else if (M == P && N == R && Q == S)
      contribution += -1.0*getRDM(I,J,M,N,Q,K,L,P,R,S, rdm );

    else if (M == P && N == R && Q == I)
      contribution += 1.0*getRDM(J,M,N,Q,S,K,L,P,R,S, rdm );

    else if (M == P && N == R && P == S)
      contribution += 1.0*getRDM(I,J,M,N,Q,K,L,Q,R,S, rdm );

    else if (M == P && N == R && P == J)
      contribution += 1.0*getRDM(I,M,N,Q,S,K,L,Q,R,S, rdm );

    else if (M == P && N == J && L == R)
      contribution += 1.0*getRDM(I,M,N,Q,S,K,P,Q,R,S, rdm );

    else if (M == P && N == J && K == S)
      contribution += 1.0*getRDM(I,M,N,Q,R,L,P,Q,R,S, rdm );

    else if (M == P && N == J && Q == S)
      contribution += -1.0*getRDM(I,M,N,Q,R,K,L,P,R,S, rdm );

    else if (M == P && N == J && Q == R)
      contribution += 1.0*getRDM(I,M,N,Q,S,K,L,P,R,S, rdm );

    else if (M == P && N == J && Q == I)
      contribution += 1.0*getRDM(M,N,Q,R,S,K,L,P,R,S, rdm );

    else if (M == P && N == J && P == S)
      contribution += 1.0*getRDM(I,M,N,Q,R,K,L,Q,R,S, rdm );

    else if (M == P && N == J && P == R)
      contribution += -1.0*getRDM(I,M,N,Q,S,K,L,Q,R,S, rdm );

    else if (M == P && N == I && L == R)
      contribution += -1.0*getRDM(J,M,N,Q,S,K,P,Q,R,S, rdm );

    else if (M == P && N == I && K == S)
      contribution += -1.0*getRDM(J,M,N,Q,R,L,P,Q,R,S, rdm );

    else if (M == P && N == I && Q == S)
      contribution += 1.0*getRDM(J,M,N,Q,R,K,L,P,R,S, rdm );

    else if (M == P && N == I && Q == R)
      contribution += -1.0*getRDM(J,M,N,Q,S,K,L,P,R,S, rdm );

    else if (M == P && N == I && P == S)
      contribution += -1.0*getRDM(J,M,N,Q,R,K,L,Q,R,S, rdm );

    else if (M == P && N == I && P == R)
      contribution += 1.0*getRDM(J,M,N,Q,S,K,L,Q,R,S, rdm );

    else if (M == P && N == I && P == J)
      contribution += 1.0*getRDM(M,N,Q,R,S,K,L,Q,R,S, rdm );

    else if (M == P && N == Q && L == R)
      contribution += -1.0*getRDM(I,J,M,N,S,K,P,Q,R,S, rdm );

    else if (M == P && N == Q && K == S)
      contribution += -1.0*getRDM(I,J,M,N,R,L,P,Q,R,S, rdm );

    else if (M == P && N == Q && Q == S)
      contribution += 1.0*getRDM(I,J,M,N,R,K,L,P,R,S, rdm );

    else if (M == P && N == Q && Q == R)
      contribution += -1.0*getRDM(I,J,M,N,S,K,L,P,R,S, rdm );

    else if (M == P && N == Q && Q == I)
      contribution += -1.0*getRDM(J,M,N,R,S,K,L,P,R,S, rdm );

    else if (M == P && N == Q && P == S)
      contribution += -1.0*getRDM(I,J,M,N,R,K,L,Q,R,S, rdm );

    else if (M == P && N == Q && P == R)
      contribution += 1.0*getRDM(I,J,M,N,S,K,L,Q,R,S, rdm );

    else if (M == P && N == Q && P == J)
      contribution += -1.0*getRDM(I,M,N,R,S,K,L,Q,R,S, rdm );

    else if (K == S && L == R)
      contribution += -1.0*getRDM(I,J,M,N,P,Q,M,N,P,Q,R,S, rdm );

    else if (Q == S && L == R)
      contribution += -1.0*getRDM(I,J,M,N,P,Q,K,M,N,P,R,S, rdm );

    else if (Q == R && K == S)
      contribution += -1.0*getRDM(I,J,M,N,P,Q,L,M,N,P,R,S, rdm );

    else if (Q == I && L == R)
      contribution += -1.0*getRDM(J,M,N,P,Q,S,K,M,N,P,R,S, rdm );

    else if (Q == I && K == S)
      contribution += -1.0*getRDM(J,M,N,P,Q,R,L,M,N,P,R,S, rdm );

    else if (P == S && L == R)
      contribution += 1.0*getRDM(I,J,M,N,P,Q,K,M,N,Q,R,S, rdm );

    else if (P == S && Q == R)
      contribution += -1.0*getRDM(I,J,M,N,P,Q,K,L,M,N,R,S, rdm );

    else if (P == S && Q == I)
      contribution += -1.0*getRDM(J,M,N,P,Q,R,K,L,M,N,R,S, rdm );

    else if (P == R && K == S)
      contribution += 1.0*getRDM(I,J,M,N,P,Q,L,M,N,Q,R,S, rdm );

    else if (P == R && Q == S)
      contribution += 1.0*getRDM(I,J,M,N,P,Q,K,L,M,N,R,S, rdm );

    else if (P == R && Q == I)
      contribution += 1.0*getRDM(J,M,N,P,Q,S,K,L,M,N,R,S, rdm );

    else if (P == J && L == R)
      contribution += -1.0*getRDM(I,M,N,P,Q,S,K,M,N,Q,R,S, rdm );

    else if (P == J && K == S)
      contribution += -1.0*getRDM(I,M,N,P,Q,R,L,M,N,Q,R,S, rdm );

    else if (P == J && Q == S)
      contribution += -1.0*getRDM(I,M,N,P,Q,R,K,L,M,N,R,S, rdm );

    else if (P == J && Q == R)
      contribution += 1.0*getRDM(I,M,N,P,Q,S,K,L,M,N,R,S, rdm );

    else if (P == J && Q == I)
      contribution += -1.0*getRDM(M,N,P,Q,R,S,K,L,M,N,R,S, rdm );

    else if (N == S && L == R)
      contribution += -1.0*getRDM(I,J,M,N,P,Q,K,M,P,Q,R,S, rdm );

    else if (N == S && Q == R)
      contribution += 1.0*getRDM(I,J,M,N,P,Q,K,L,M,P,R,S, rdm );

    else if (N == S && Q == I)
      contribution += 1.0*getRDM(J,M,N,P,Q,R,K,L,M,P,R,S, rdm );

    else if (N == S && P == R)
      contribution += -1.0*getRDM(I,J,M,N,P,Q,K,L,M,Q,R,S, rdm );

    else if (N == S && P == J)
      contribution += 1.0*getRDM(I,M,N,P,Q,R,K,L,M,Q,R,S, rdm );

    else if (N == R && K == S)
      contribution += -1.0*getRDM(I,J,M,N,P,Q,L,M,P,Q,R,S, rdm );

    else if (N == R && Q == S)
      contribution += -1.0*getRDM(I,J,M,N,P,Q,K,L,M,P,R,S, rdm );

    else if (N == R && Q == I)
      contribution += -1.0*getRDM(J,M,N,P,Q,S,K,L,M,P,R,S, rdm );

    else if (N == R && P == S)
      contribution += 1.0*getRDM(I,J,M,N,P,Q,K,L,M,Q,R,S, rdm );

    else if (N == R && P == J)
      contribution += -1.0*getRDM(I,M,N,P,Q,S,K,L,M,Q,R,S, rdm );

    else if (N == J && L == R)
      contribution += 1.0*getRDM(I,M,N,P,Q,S,K,M,P,Q,R,S, rdm );

    else if (N == J && K == S)
      contribution += 1.0*getRDM(I,M,N,P,Q,R,L,M,P,Q,R,S, rdm );

    else if (N == J && Q == S)
      contribution += 1.0*getRDM(I,M,N,P,Q,R,K,L,M,P,R,S, rdm );

    else if (N == J && Q == R)
      contribution += -1.0*getRDM(I,M,N,P,Q,S,K,L,M,P,R,S, rdm );

    else if (N == J && Q == I)
      contribution += 1.0*getRDM(M,N,P,Q,R,S,K,L,M,P,R,S, rdm );

    else if (N == J && P == S)
      contribution += -1.0*getRDM(I,M,N,P,Q,R,K,L,M,Q,R,S, rdm );

    else if (N == J && P == R)
      contribution += 1.0*getRDM(I,M,N,P,Q,S,K,L,M,Q,R,S, rdm );

    else if (N == I && L == R)
      contribution += -1.0*getRDM(J,M,N,P,Q,S,K,M,P,Q,R,S, rdm );

    else if (N == I && K == S)
      contribution += -1.0*getRDM(J,M,N,P,Q,R,L,M,P,Q,R,S, rdm );

    else if (N == I && Q == S)
      contribution += -1.0*getRDM(J,M,N,P,Q,R,K,L,M,P,R,S, rdm );

    else if (N == I && Q == R)
      contribution += 1.0*getRDM(J,M,N,P,Q,S,K,L,M,P,R,S, rdm );

    else if (N == I && P == S)
      contribution += 1.0*getRDM(J,M,N,P,Q,R,K,L,M,Q,R,S, rdm );

    else if (N == I && P == R)
      contribution += -1.0*getRDM(J,M,N,P,Q,S,K,L,M,Q,R,S, rdm );

    else if (N == I && P == J)
      contribution += 1.0*getRDM(M,N,P,Q,R,S,K,L,M,Q,R,S, rdm );

    else if (N == Q && L == R)
      contribution += 1.0*getRDM(I,J,M,N,P,S,K,M,P,Q,R,S, rdm );

    else if (N == Q && K == S)
      contribution += 1.0*getRDM(I,J,M,N,P,R,L,M,P,Q,R,S, rdm );

    else if (N == Q && Q == S)
      contribution += 1.0*getRDM(I,J,M,N,P,R,K,L,M,P,R,S, rdm );

    else if (N == Q && Q == R)
      contribution += -1.0*getRDM(I,J,M,N,P,S,K,L,M,P,R,S, rdm );

    else if (N == Q && Q == I)
      contribution += 1.0*getRDM(J,M,N,P,R,S,K,L,M,P,R,S, rdm );

    else if (N == Q && P == S)
      contribution += -1.0*getRDM(I,J,M,N,P,R,K,L,M,Q,R,S, rdm );

    else if (N == Q && P == R)
      contribution += 1.0*getRDM(I,J,M,N,P,S,K,L,M,Q,R,S, rdm );

    else if (N == Q && P == J)
      contribution += 1.0*getRDM(I,M,N,P,R,S,K,L,M,Q,R,S, rdm );

    else if (N == P && L == R)
      contribution += -1.0*getRDM(I,J,M,N,Q,S,K,M,P,Q,R,S, rdm );

    else if (N == P && K == S)
      contribution += -1.0*getRDM(I,J,M,N,Q,R,L,M,P,Q,R,S, rdm );

    else if (N == P && Q == S)
      contribution += -1.0*getRDM(I,J,M,N,Q,R,K,L,M,P,R,S, rdm );

    else if (N == P && Q == R)
      contribution += 1.0*getRDM(I,J,M,N,Q,S,K,L,M,P,R,S, rdm );

    else if (N == P && Q == I)
      contribution += -1.0*getRDM(J,M,N,Q,R,S,K,L,M,P,R,S, rdm );

    else if (N == P && P == S)
      contribution += 1.0*getRDM(I,J,M,N,Q,R,K,L,M,Q,R,S, rdm );

    else if (N == P && P == R)
      contribution += -1.0*getRDM(I,J,M,N,Q,S,K,L,M,Q,R,S, rdm );

    else if (N == P && P == J)
      contribution += -1.0*getRDM(I,M,N,Q,R,S,K,L,M,Q,R,S, rdm );

    else if (M == S && L == R)
      contribution += 1.0*getRDM(I,J,M,N,P,Q,K,N,P,Q,R,S, rdm );

    else if (M == S && Q == R)
      contribution += -1.0*getRDM(I,J,M,N,P,Q,K,L,N,P,R,S, rdm );

    else if (M == S && Q == I)
      contribution += -1.0*getRDM(J,M,N,P,Q,R,K,L,N,P,R,S, rdm );

    else if (M == S && P == R)
      contribution += 1.0*getRDM(I,J,M,N,P,Q,K,L,N,Q,R,S, rdm );

    else if (M == S && P == J)
      contribution += -1.0*getRDM(I,M,N,P,Q,R,K,L,N,Q,R,S, rdm );

    else if (M == S && N == R)
      contribution += -1.0*getRDM(I,J,M,N,P,Q,K,L,P,Q,R,S, rdm );

    else if (M == S && N == J)
      contribution += 1.0*getRDM(I,M,N,P,Q,R,K,L,P,Q,R,S, rdm );

    else if (M == S && N == I)
      contribution += -1.0*getRDM(J,M,N,P,Q,R,K,L,P,Q,R,S, rdm );

    else if (M == S && N == Q)
      contribution += 1.0*getRDM(I,J,M,N,P,R,K,L,P,Q,R,S, rdm );

    else if (M == S && N == P)
      contribution += -1.0*getRDM(I,J,M,N,Q,R,K,L,P,Q,R,S, rdm );

    else if (M == R && K == S)
      contribution += 1.0*getRDM(I,J,M,N,P,Q,L,N,P,Q,R,S, rdm );

    else if (M == R && Q == S)
      contribution += 1.0*getRDM(I,J,M,N,P,Q,K,L,N,P,R,S, rdm );

    else if (M == R && Q == I)
      contribution += 1.0*getRDM(J,M,N,P,Q,S,K,L,N,P,R,S, rdm );

    else if (M == R && P == S)
      contribution += -1.0*getRDM(I,J,M,N,P,Q,K,L,N,Q,R,S, rdm );

    else if (M == R && P == J)
      contribution += 1.0*getRDM(I,M,N,P,Q,S,K,L,N,Q,R,S, rdm );

    else if (M == R && N == S)
      contribution += 1.0*getRDM(I,J,M,N,P,Q,K,L,P,Q,R,S, rdm );

    else if (M == R && N == J)
      contribution += -1.0*getRDM(I,M,N,P,Q,S,K,L,P,Q,R,S, rdm );

    else if (M == R && N == I)
      contribution += 1.0*getRDM(J,M,N,P,Q,S,K,L,P,Q,R,S, rdm );

    else if (M == R && N == Q)
      contribution += -1.0*getRDM(I,J,M,N,P,S,K,L,P,Q,R,S, rdm );

    else if (M == R && N == P)
      contribution += 1.0*getRDM(I,J,M,N,Q,S,K,L,P,Q,R,S, rdm );

    else if (M == J && L == R)
      contribution += -1.0*getRDM(I,M,N,P,Q,S,K,N,P,Q,R,S, rdm );

    else if (M == J && K == S)
      contribution += -1.0*getRDM(I,M,N,P,Q,R,L,N,P,Q,R,S, rdm );

    else if (M == J && Q == S)
      contribution += -1.0*getRDM(I,M,N,P,Q,R,K,L,N,P,R,S, rdm );

    else if (M == J && Q == R)
      contribution += 1.0*getRDM(I,M,N,P,Q,S,K,L,N,P,R,S, rdm );

    else if (M == J && Q == I)
      contribution += -1.0*getRDM(M,N,P,Q,R,S,K,L,N,P,R,S, rdm );

    else if (M == J && P == S)
      contribution += 1.0*getRDM(I,M,N,P,Q,R,K,L,N,Q,R,S, rdm );

    else if (M == J && P == R)
      contribution += -1.0*getRDM(I,M,N,P,Q,S,K,L,N,Q,R,S, rdm );

    else if (M == J && N == S)
      contribution += -1.0*getRDM(I,M,N,P,Q,R,K,L,P,Q,R,S, rdm );

    else if (M == J && N == R)
      contribution += 1.0*getRDM(I,M,N,P,Q,S,K,L,P,Q,R,S, rdm );

    else if (M == J && N == I)
      contribution += -1.0*getRDM(M,N,P,Q,R,S,K,L,P,Q,R,S, rdm );

    else if (M == J && N == Q)
      contribution += -1.0*getRDM(I,M,N,P,R,S,K,L,P,Q,R,S, rdm );

    else if (M == J && N == P)
      contribution += 1.0*getRDM(I,M,N,Q,R,S,K,L,P,Q,R,S, rdm );

    else if (M == I && L == R)
      contribution += 1.0*getRDM(J,M,N,P,Q,S,K,N,P,Q,R,S, rdm );

    else if (M == I && K == S)
      contribution += 1.0*getRDM(J,M,N,P,Q,R,L,N,P,Q,R,S, rdm );

    else if (M == I && Q == S)
      contribution += 1.0*getRDM(J,M,N,P,Q,R,K,L,N,P,R,S, rdm );

    else if (M == I && Q == R)
      contribution += -1.0*getRDM(J,M,N,P,Q,S,K,L,N,P,R,S, rdm );

    else if (M == I && P == S)
      contribution += -1.0*getRDM(J,M,N,P,Q,R,K,L,N,Q,R,S, rdm );

    else if (M == I && P == R)
      contribution += 1.0*getRDM(J,M,N,P,Q,S,K,L,N,Q,R,S, rdm );

    else if (M == I && P == J)
      contribution += -1.0*getRDM(M,N,P,Q,R,S,K,L,N,Q,R,S, rdm );

    else if (M == I && N == S)
      contribution += 1.0*getRDM(J,M,N,P,Q,R,K,L,P,Q,R,S, rdm );

    else if (M == I && N == R)
      contribution += -1.0*getRDM(J,M,N,P,Q,S,K,L,P,Q,R,S, rdm );

    else if (M == I && N == J)
      contribution += 1.0*getRDM(M,N,P,Q,R,S,K,L,P,Q,R,S, rdm );

    else if (M == I && N == Q)
      contribution += 1.0*getRDM(J,M,N,P,R,S,K,L,P,Q,R,S, rdm );

    else if (M == I && N == P)
      contribution += -1.0*getRDM(J,M,N,Q,R,S,K,L,P,Q,R,S, rdm );

    else if (M == Q && L == R)
      contribution += -1.0*getRDM(I,J,M,N,P,S,K,N,P,Q,R,S, rdm );

    else if (M == Q && K == S)
      contribution += -1.0*getRDM(I,J,M,N,P,R,L,N,P,Q,R,S, rdm );

    else if (M == Q && Q == S)
      contribution += -1.0*getRDM(I,J,M,N,P,R,K,L,N,P,R,S, rdm );

    else if (M == Q && Q == R)
      contribution += 1.0*getRDM(I,J,M,N,P,S,K,L,N,P,R,S, rdm );

    else if (M == Q && Q == I)
      contribution += -1.0*getRDM(J,M,N,P,R,S,K,L,N,P,R,S, rdm );

    else if (M == Q && P == S)
      contribution += 1.0*getRDM(I,J,M,N,P,R,K,L,N,Q,R,S, rdm );

    else if (M == Q && P == R)
      contribution += -1.0*getRDM(I,J,M,N,P,S,K,L,N,Q,R,S, rdm );

    else if (M == Q && P == J)
      contribution += -1.0*getRDM(I,M,N,P,R,S,K,L,N,Q,R,S, rdm );

    else if (M == Q && N == S)
      contribution += -1.0*getRDM(I,J,M,N,P,R,K,L,P,Q,R,S, rdm );

    else if (M == Q && N == R)
      contribution += 1.0*getRDM(I,J,M,N,P,S,K,L,P,Q,R,S, rdm );

    else if (M == Q && N == J)
      contribution += 1.0*getRDM(I,M,N,P,R,S,K,L,P,Q,R,S, rdm );

    else if (M == Q && N == I)
      contribution += -1.0*getRDM(J,M,N,P,R,S,K,L,P,Q,R,S, rdm );

    else if (M == Q && N == P)
      contribution += -1.0*getRDM(I,J,M,N,R,S,K,L,P,Q,R,S, rdm );

    else if (M == P && L == R)
      contribution += 1.0*getRDM(I,J,M,N,Q,S,K,N,P,Q,R,S, rdm );

    else if (M == P && K == S)
      contribution += 1.0*getRDM(I,J,M,N,Q,R,L,N,P,Q,R,S, rdm );

    else if (M == P && Q == S)
      contribution += 1.0*getRDM(I,J,M,N,Q,R,K,L,N,P,R,S, rdm );

    else if (M == P && Q == R)
      contribution += -1.0*getRDM(I,J,M,N,Q,S,K,L,N,P,R,S, rdm );

    else if (M == P && Q == I)
      contribution += 1.0*getRDM(J,M,N,Q,R,S,K,L,N,P,R,S, rdm );

    else if (M == P && P == S)
      contribution += -1.0*getRDM(I,J,M,N,Q,R,K,L,N,Q,R,S, rdm );

    else if (M == P && P == R)
      contribution += 1.0*getRDM(I,J,M,N,Q,S,K,L,N,Q,R,S, rdm );

    else if (M == P && P == J)
      contribution += 1.0*getRDM(I,M,N,Q,R,S,K,L,N,Q,R,S, rdm );

    else if (M == P && N == S)
      contribution += 1.0*getRDM(I,J,M,N,Q,R,K,L,P,Q,R,S, rdm );

    else if (M == P && N == R)
      contribution += -1.0*getRDM(I,J,M,N,Q,S,K,L,P,Q,R,S, rdm );

    else if (M == P && N == J)
      contribution += -1.0*getRDM(I,M,N,Q,R,S,K,L,P,Q,R,S, rdm );

    else if (M == P && N == I)
      contribution += 1.0*getRDM(J,M,N,Q,R,S,K,L,P,Q,R,S, rdm );

    else if (M == P && N == Q)
      contribution += 1.0*getRDM(I,J,M,N,R,S,K,L,P,Q,R,S, rdm );

    else if (L == R)
      contribution += 1.0*getRDM(I,J,M,N,P,Q,S,K,M,N,P,Q,R,S, rdm );

    else if (K == S)
      contribution += 1.0*getRDM(I,J,M,N,P,Q,R,L,M,N,P,Q,R,S, rdm );

    else if (Q == S)
      contribution += -1.0*getRDM(I,J,M,N,P,Q,R,K,L,M,N,P,R,S, rdm );

    else if (Q == R)
      contribution += 1.0*getRDM(I,J,M,N,P,Q,S,K,L,M,N,P,R,S, rdm );

    else if (Q == I)
      contribution += 1.0*getRDM(J,M,N,P,Q,R,S,K,L,M,N,P,R,S, rdm );

    else if (P == S)
      contribution += 1.0*getRDM(I,J,M,N,P,Q,R,K,L,M,N,Q,R,S, rdm );

    else if (P == R)
      contribution += -1.0*getRDM(I,J,M,N,P,Q,S,K,L,M,N,Q,R,S, rdm );

    else if (P == J)
      contribution += 1.0*getRDM(I,M,N,P,Q,R,S,K,L,M,N,Q,R,S, rdm );

    else if (N == S)
      contribution += -1.0*getRDM(I,J,M,N,P,Q,R,K,L,M,P,Q,R,S, rdm );

    else if (N == R)
      contribution += 1.0*getRDM(I,J,M,N,P,Q,S,K,L,M,P,Q,R,S, rdm );

    else if (N == J)
      contribution += -1.0*getRDM(I,M,N,P,Q,R,S,K,L,M,P,Q,R,S, rdm );

    else if (N == I)
      contribution += 1.0*getRDM(J,M,N,P,Q,R,S,K,L,M,P,Q,R,S, rdm );

    else if (N == Q)
      contribution += -1.0*getRDM(I,J,M,N,P,R,S,K,L,M,P,Q,R,S, rdm );

    else if (N == P)
      contribution += 1.0*getRDM(I,J,M,N,Q,R,S,K,L,M,P,Q,R,S, rdm );

    else if (M == S)
      contribution += 1.0*getRDM(I,J,M,N,P,Q,R,K,L,N,P,Q,R,S, rdm );

    else if (M == R)
      contribution += -1.0*getRDM(I,J,M,N,P,Q,S,K,L,N,P,Q,R,S, rdm );

    else if (M == J)
      contribution += 1.0*getRDM(I,M,N,P,Q,R,S,K,L,N,P,Q,R,S, rdm );

    else if (M == I)
      contribution += -1.0*getRDM(J,M,N,P,Q,R,S,K,L,N,P,Q,R,S, rdm );

    else if (M == Q)
      contribution += 1.0*getRDM(I,J,M,N,P,R,S,K,L,N,P,Q,R,S, rdm );

    else if (M == P)
      contribution += -1.0*getRDM(I,J,M,N,Q,R,S,K,L,N,P,Q,R,S, rdm );

    else 
      contribution += -1.0*getRDM(I,J,M,N,P,Q,R,S,K,L,M,N,P,Q,R,S, rdm );

    return contribution;
    
  }
  
  complexT calcTerm1(int P, int Q, int I, int K, MatrixXcT& rdm) {
    complexT contribution = complexT(0.,0.);
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

  complexT calcTerm1(int P, int Q, int I, int J, int K, int L, MatrixXcT& rdm) {
    if (K == -1 && L == -1)
      return calcTerm1(P, Q, I, J, rdm);
    complexT contribution = complexT(0.,0.);
  
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


  complexT calcTerm2(int P, int Q, int I, int K, int R, int S, MatrixXcT& rdm) {
    complexT contribution = complexT(0.,0.);
  
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


  complexT calcTerm2(int P, int Q, int I, int J, int K, int L, int R, int S, MatrixXcT& rdm){
    if (K == -1 && L == -1)
      calcTerm2(P, Q, I, J, R, S, rdm);
    complexT contribution(0.0,0.);
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
  /*
    complex<T> getRDM(int a, int b, MatrixXcT& rdm);
    complex<T> getRDM(int a, int b, int c, int d, MatrixXcT& rdm);
    complex<T> getRDM(int a, int b, int c, int d, int e, int f, MatrixXcT& rdm);
    complex<T> getRDM(int a, int b, int c, int d, int e, int f, int g, int h, MatrixXcT& rdm);
    complex<T> getRDM(int a, int b, int c, int d, int e, int f, int g, int h, int i, int j, MatrixXcT& rdm);
    complex<T> getRDM(int a, int b, int c, int d, int e, int f, int g, int h, int i, int j, int k, int l, MatrixXcT& rdm);

    complex<T> calcTerm1(int P, int Q, int I, int K, MatrixXcT& rdm);
    complex<T> calcTerm1(int P, int Q, int I, int J, int K, int L, MatrixXcT& rdm);
    complex<T> calcTerm2(int P, int Q, int I, int K, int R, int S, MatrixXcT& rdm);
    complex<T> calcTerm2(int P, int Q, int I, int J, int K, int L, int R, int S, MatrixXcT& rdm);
  */
};

//extern template calcRDM<complex<double>>;
//extern template calcRDM<Complex<stan::math::var>>;
