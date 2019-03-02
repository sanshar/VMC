#ifndef READSLATER_H
#define READSLATER_H

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>
#include <cassert>
#include <exception>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "slaterBasis.h"
#include <Eigen/Dense>
#include <map>

struct slaterParser {
  string fileName;
  boost::property_tree::ptree pt;
  map<string, int> angularMomStoI;
  map<int, string> angularMomItoS;
  static vector<string> AtomSymbols;
  
  slaterParser(string pfile) : fileName(pfile)
  {
    angularMomStoI["s"] = 0;
    angularMomStoI["p"] = 1;
    angularMomStoI["d"] = 2;
    angularMomStoI["f"] = 3;
    angularMomStoI["g"] = 4;
    angularMomStoI["h"] = 5;
    angularMomStoI["i"] = 6;

    angularMomItoS[0] = "S";
    angularMomItoS[1] = "P";
    angularMomItoS[2] = "D";
    angularMomItoS[3] = "F";
    angularMomItoS[4] = "G";
    angularMomItoS[5] = "H";
    angularMomItoS[6] = "I";

    std::ifstream file(fileName.c_str());
    std::stringstream ss;
    ss << file.rdbuf();
    boost::property_tree::read_json(ss, pt);
  };  

  void readBasis(vector<string>& atomList,
                 vector<Vector3d>& atomCoord,
                 slaterBasis& STO) {
    
    //loop over all the atoms
    //for (auto it = atomList.begin(); it != atomList.end(); it++) {
    for (int i=0; i<atomList.size(); i++) {
      string atom = atomList[i];
      auto v = pt.get_child(atom);
      
      slaterBasisOnAtom shell;
      shell.coord = atomCoord[i];
      shell.norbs = 0;
      
      auto data = v.begin()->second;

      auto quant = data.get_child("quant");
      for (auto nl = quant.begin(); nl != quant.end(); nl++) {
        string N = nl->second.get_value<string>().substr(0,1),
            L = nl->second.get_value<string>().substr(1,1);
        shell.NL.push_back(atoi(N.c_str()));
        shell.NL.push_back(angularMomStoI[L]);

        int l = *shell.NL.rbegin();
        shell.norbs += (l+1)*(l+2)/2;
      }
      
      auto prim = data.get_child("prim");
      int index = 0;
      for (auto ex = prim.begin(); ex!= prim.end(); ex++){
        double eta = atof(ex->second.get_value<string>().c_str());
        shell.exponents.push_back(eta);
        shell.radialNorm.push_back(STOradialNorm(eta, shell.NL[2*index]));
        index++;
      }
      
      STO.atomicBasis.push_back(shell);
    }
  };
  
};


#endif
