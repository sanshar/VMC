#include <algorithm>
#include "Residuals.h"
#include "Determinants.h"
#include "global.h"
#include "integral.h"
#include "input.h"
#include "calcRDM.h"
#include <functional>

using namespace std;
using namespace std::placeholders;



int index(int I, int J) {
  return max(I,J)*(max(I,J)+1)/2 + min(I,J);
}

int ABAB(const int &orb) {
  const int& norbs = Determinant::norbs;
  return (orb%norbs)* 2 + orb/norbs;
};
  


void ConstructRedundantJastrowMap(vector<pair<int,int>>& NonRedundantMap) {
  int norbs = Determinant::norbs;

  if (schd.wavefunctionType == "JastrowSlater") {
    for (int i=0; i<2*norbs; i++)
      for (int j=0; j<i; j++) {
        //if (j == 0 || (i==2 && j == 1) || (i==3 && j==1))
        if (j == 0 || (i==2 && j == 1) )
          continue;
        else
          NonRedundantMap.push_back(make_pair(i, j));
      }
  }
  else if (schd.wavefunctionType == "GutzwillerSlater") {
    for (int i=0; i<norbs; i++)
      NonRedundantMap.push_back(make_pair(i+norbs, i));
  }
}



