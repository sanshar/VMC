/*
  Developed by Sandeep Sharma
  Copyright (c) 2017, Sandeep Sharma
  
  This file is part of DICE.
  
  This program is free software: you can redistribute it and/or modify it under the terms
  of the GNU General Public License as published by the Free Software Foundation, 
  either version 3 of the License, or (at your option) any later version.
  
  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
  
  See the GNU General Public License for more details.
  
  You should have received a copy of the GNU General Public License along with this program. 
  If not, see <http://www.gnu.org/licenses/>.
*/
#include "input.h"
#include "CPS.h"
#include "global.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>

using namespace Eigen;
using namespace boost;
using namespace std;


void readInput(string input, schedule& schd, bool print) {
  if (print) {
  cout << "**************************************************************"<<endl;
  cout << "Input file  :"<<endl;
  cout << "**************************************************************"<<endl;
  }

  ifstream dump(input.c_str());

  schd.deterministic          = false;
  schd.restart                = false;
  schd.davidsonPrecondition   = false;
  schd.diisSize               = 5;
  schd.maxIter                = 50;
  schd.gradientFactor         = 0.001;
  schd.mingradientFactor      = 0.00001;
  schd.m                      = rmsprop;
  schd.stochasticIter         = 1e4;
  schd.integralSampleSize     = 10;
  schd.momentum               = 0.9;
  schd.momentumDecay          = 0.001;
  schd.decay                  = 0.9;
  schd.learningEpoch          = 10;
  schd.seed                   = getTime();
  schd.PTlambda               = 0.5;
  schd.epsilon                = 1.e-7;
  schd.singleProbability      = 0.5;
  schd.doubleProbability      = 0.5;
  schd.screen                 = 1.e-8;

  while (dump.good()) {

    std::string
      Line;
    std::getline(dump, Line);
    trim(Line);
    if (print) cout <<Line<<endl;

    vector<string> tok;
    boost::split(tok, Line, is_any_of(", \t\n"), token_compress_on);
    string ArgName = *tok.begin();

    //if (dump.eof())
    //break;
    if (!ArgName.empty() && (boost::iequals(tok[0].substr(0,1), "#"))) continue;
    if (ArgName.empty()) continue;

    if (boost::iequals(ArgName,       "restart"       ))
      schd.restart = true;

    else if (boost::iequals(ArgName,  "deterministic" ))
      schd.deterministic = true;

    else if (boost::iequals(ArgName,  "adam"          ))
      schd.m = adam;

    else if (boost::iequals(ArgName,  "sgd"           ))
      schd.m = sgd;

    else if (boost::iequals(ArgName,  "nestorov"      ))
      schd.m = nestorov;

    else if (boost::iequals(ArgName,  "rmsprop"       ))
      schd.m = rmsprop;

    else if (boost::iequals(ArgName,  "ptlambda"      ))
      schd.PTlambda = atof(tok[1].c_str());

    else if (boost::iequals(ArgName,  "amsgrad"       ))
      schd.m = amsgrad;

    else if (boost::iequals(ArgName,  "tol"           ))
      schd.tol = atof(tok[1].c_str());

    else if (boost::iequals(ArgName,  "screentol"     ))
      schd.screen = atof(tok[1].c_str());

    else if (boost::iequals(ArgName,  "momentum"      ))
      schd.momentum = atof(tok[1].c_str());

    else if (boost::iequals(ArgName,  "momentumDecay" ))
      schd.momentumDecay = atof(tok[1].c_str());

    else if (boost::iequals(ArgName,  "epsilon"       ))
      schd.epsilon = atof(tok[1].c_str());

    else if (boost::iequals(ArgName,  "seed"          ))
      schd.seed = atof(tok[1].c_str());

    else if (boost::iequals(ArgName,  "decay"         ))
      schd.decay = atof(tok[1].c_str());

    else if (boost::iequals(ArgName,  "learningepoch" ))
      schd.learningEpoch = atoi(tok[1].c_str());

    else if (boost::iequals(ArgName,  "stochasticiter"  ))
      schd.stochasticIter = atoi(tok[1].c_str());

    else if (boost::iequals(ArgName,  "integralsamplesize"))
      schd.integralSampleSize = atoi(tok[1].c_str());

    else if (boost::iequals(ArgName,  "gradientFactor"  ))
      schd.gradientFactor = atof(tok[1].c_str());

    else if (boost::iequals(ArgName,  "mingradientFactor"))
      schd.mingradientFactor = atof(tok[1].c_str());

    else if (boost::iequals(ArgName,  "correlator"       )) {
      int siteSize = atoi(tok[1].c_str());
      schd.correlatorFiles[siteSize] = tok[2];
    }

    else if (boost::iequals(ArgName,  "Precondition"     )) {
      schd.davidsonPrecondition = true;
    }

    else if (boost::iequals(ArgName,  "diisSize"          )) {
      schd.diisSize = atoi(tok[1].c_str());
    }

    else if (boost::iequals(ArgName,  "printLevel"        )) {
      schd.printLevel = atoi(tok[1].c_str());
    }

    else if (boost::iequals(ArgName,  "maxiter"           )) {
      schd.maxIter = atoi(tok[1].c_str());
    }

    else {
      cout << "cannot read option "<<ArgName<<endl;
      exit(0);
    }

  }


}


void readCorrelator(std::string input, int correlatorSize,
		    std::vector<CPS>& correlators) {
  ifstream dump(input.c_str());

  while (dump.good()) {

    std::string
      Line;
    std::getline(dump, Line);
    trim(Line);
    vector<string> tok;
    boost::split(tok, Line, is_any_of(", \t\n"), token_compress_on);

    string ArgName = *tok.begin();

    //if (dump.eof())
    //break;
    if (!ArgName.empty() && (boost::iequals(tok[0].substr(0,1), "#"))) continue;
    if (ArgName.empty()) continue;
    
    if (tok.size() != correlatorSize) {
      cout << "Something wrong in line : "<<Line<<endl;
      exit(0);
    }

    vector<int> asites, bsites;
    for (int i=0; i<correlatorSize; i++) {
      int site = atoi(tok[i].c_str());
      asites.push_back(site);
      bsites.push_back(site);
    }
    correlators.push_back(CPS(asites, bsites));
  }
}


void readHF(MatrixXd& Hfmatrix) {

  ifstream dump("hf.txt");
  for (int i=0; i<Hfmatrix.rows(); i++)
  for (int j=0; j<Hfmatrix.rows(); j++)
    dump >> Hfmatrix(i,j);
}
