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
#include "Determinants.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#include "readSlater.h"
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>


#ifndef SERIAL
#include "mpi.h"
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif

using namespace Eigen;
using namespace boost;
using namespace std;


void readInput(string inputFile, schedule& schd, bool print) {
  if (commrank == 0) {
    
    property_tree::iptree input;
    property_tree::read_json(inputFile, input);
    
    if (print)	{
      cout << "**************************************************************" << endl;
      cout << "Input file  :" << endl;
      cout << "**************************************************************" << endl;
      //print input file
      stringstream ss;
      property_tree::json_parser::write_json(ss, input);
      cout << ss.str() << endl;
    }

    schd.nciCore = input.get("system.numCore", 0);                  // TODO: rename these because active spaces are also used without ci
    schd.nciAct = input.get("system.numAct", -1);

    //check for realspace block, containing only system info, wave function info still in that block
    optional< property_tree::iptree& > realspace = input.get_child_optional("realspace");
    if (realspace) {
      //basis
      string basis = input.get("realspace.basis", "sto");
      if (basis == "gto") {
        schd.walkerBasis = REALSPACEGTO;
        schd.basis = boost::shared_ptr<Basis>(new gaussianBasis);
        schd.basis->read();
        readGeometry(schd.Ncoords, schd.Ncharge, schd.uniqueAtoms, schd.uniqueAtomsMap, schd.Nbasis, schd.NSbasis, schd.NPbasis, dynamic_cast<gaussianBasis&>(*schd.basis));
        schd.gBasis = boost::shared_ptr<Basis>(new gaussianBasis);
        schd.gBasis->read();
      }
      if (basis == "sto") {
        schd.walkerBasis = REALSPACESTO;
        //read gaussian basis just to read the nuclear charge and coordinates
        schd.gBasis = boost::shared_ptr<Basis>(new gaussianBasis);
        schd.gBasis->read();
        readGeometry(schd.Ncoords, schd.Ncharge, schd.uniqueAtoms, schd.uniqueAtomsMap, schd.Nbasis, schd.NSbasis, schd.NPbasis, dynamic_cast<gaussianBasis&>(*schd.gBasis));
        schd.basis = boost::shared_ptr<Basis>(new slaterBasis);
        map<string, Vector3d> atomList;
        for (int i=0; i<schd.Ncoords.size(); i++) {
          dynamic_cast<slaterBasis*>(&(*schd.basis))->atomName.push_back(slaterParser::AtomSymbols[schd.Ncharge[i]]);
          dynamic_cast<slaterBasis*>(&(*schd.basis))->atomCoord.push_back(schd.Ncoords[i]);
        }
        //dynamic_cast<slaterBasis*>(&(*schd.basis))->atomList = atomList;
        schd.basis->read();
      }
      
      //electrons, orbs
      schd.nalpha = input.get("realspace.nalpha", -1);
      schd.nbeta = input.get("realspace.nbeta", -1);
      schd.norbs = input.get("realspace.norbs", 0);
      
      //pseudopotential
      schd.pseudo = boost::shared_ptr<Pseudopotential>(new Pseudopotential);
      schd.nGrid = input.get("realspace.nGrid", 5);
      schd.pCutOff = input.get("realspace.pseudoCutOff", 1.0e-8);
      schd.pQuad = input.get("realspace.pseudo", 4);
      if (schd.pQuad == 4)
      {
        //sample 4 vertices of tetrahedral
        double a = 1.0 / std::sqrt(3.0);
        schd.Q.push_back(Vector3d(a, a, a));
        schd.Q.push_back(Vector3d(a, -a, -a));
        schd.Q.push_back(Vector3d(-a, a, -a));
        schd.Q.push_back(Vector3d(-a, -a, a));

        schd.Qwt.push_back(1.0 / 4.0);
        schd.Qwt.push_back(1.0 / 4.0);
        schd.Qwt.push_back(1.0 / 4.0);
        schd.Qwt.push_back(1.0 / 4.0);
      }
      else if (schd.pQuad == 6)
      {
        //sample 6 vertices of octahedral
        schd.Q.push_back(Vector3d(1.0, 0.0, 0.0));
        schd.Q.push_back(Vector3d(-1.0, 0.0, 0.0));
        schd.Q.push_back(Vector3d(0.0, 1.0, 0.0));
        schd.Q.push_back(Vector3d(0.0, -1.0, 0.0));
        schd.Q.push_back(Vector3d(0.0, 0.0, 1.0));
        schd.Q.push_back(Vector3d(0.0, 0.0, -1.0));

        schd.Qwt.push_back(1.0 / 6.0);
        schd.Qwt.push_back(1.0 / 6.0);
        schd.Qwt.push_back(1.0 / 6.0);
        schd.Qwt.push_back(1.0 / 6.0);
        schd.Qwt.push_back(1.0 / 6.0);
        schd.Qwt.push_back(1.0 / 6.0);
      }
      else if (schd.pQuad == 12)
      {
        //sample 12 vertices of icosahedral
        double lambda = std::sqrt((5.0 - std::sqrt(5.0)) / 10.0);
        double roh = std::sqrt((5.0 + std::sqrt(5.0)) / 10.0);
        schd.Q.push_back(Vector3d(0.0, lambda, roh));
        schd.Q.push_back(Vector3d(0.0, -lambda, roh));
        schd.Q.push_back(Vector3d(0.0, lambda, -roh));
        schd.Q.push_back(Vector3d(0.0, -lambda, -roh));

        schd.Q.push_back(Vector3d(lambda, 0.0, roh));
        schd.Q.push_back(Vector3d(-lambda, 0.0, roh));
        schd.Q.push_back(Vector3d(lambda, 0.0, -roh));
        schd.Q.push_back(Vector3d(-lambda, 0.0, -roh));

        schd.Q.push_back(Vector3d(lambda, roh, 0.0));
        schd.Q.push_back(Vector3d(-lambda, roh, 0.0));
        schd.Q.push_back(Vector3d(lambda, -roh, 0.0));
        schd.Q.push_back(Vector3d(-lambda, -roh, 0.0)); 

        schd.Qwt.push_back(1.0 / 12.0);
        schd.Qwt.push_back(1.0 / 12.0);
        schd.Qwt.push_back(1.0 / 12.0);
        schd.Qwt.push_back(1.0 / 12.0);
        schd.Qwt.push_back(1.0 / 12.0);
        schd.Qwt.push_back(1.0 / 12.0);

        schd.Qwt.push_back(1.0 / 12.0);
        schd.Qwt.push_back(1.0 / 12.0);
        schd.Qwt.push_back(1.0 / 12.0);
        schd.Qwt.push_back(1.0 / 12.0);
        schd.Qwt.push_back(1.0 / 12.0);
        schd.Qwt.push_back(1.0 / 12.0);
      }
      else if (schd.pQuad == 18)
      {
        schd.Q.push_back(Vector3d(1.0, 0.0, 0.0));
        schd.Q.push_back(Vector3d(-1.0, 0.0, 0.0));
        schd.Q.push_back(Vector3d(0.0, 1.0, 0.0));
        schd.Q.push_back(Vector3d(0.0, -1.0, 0.0));
        schd.Q.push_back(Vector3d(0.0, 0.0, 1.0));
        schd.Q.push_back(Vector3d(0.0, 0.0, -1.0));

        schd.Qwt.push_back(1.0 / 30.0);
        schd.Qwt.push_back(1.0 / 30.0);
        schd.Qwt.push_back(1.0 / 30.0);
        schd.Qwt.push_back(1.0 / 30.0);
        schd.Qwt.push_back(1.0 / 30.0);
        schd.Qwt.push_back(1.0 / 30.0);

        double p = 1.0 / std::sqrt(2);
        schd.Q.push_back(Vector3d(p, p, 0.0));
        schd.Q.push_back(Vector3d(-p, p, 0.0));
        schd.Q.push_back(Vector3d(p, -p, 0.0));
        schd.Q.push_back(Vector3d(-p, -p, 0.0)); 

        schd.Q.push_back(Vector3d(0.0, p, p));
        schd.Q.push_back(Vector3d(0.0, -p, p));
        schd.Q.push_back(Vector3d(0.0, p, -p));
        schd.Q.push_back(Vector3d(0.0, -p, -p));

        schd.Q.push_back(Vector3d(p, 0.0, p));
        schd.Q.push_back(Vector3d(-p, 0.0, p));
        schd.Q.push_back(Vector3d(p, 0.0, -p));
        schd.Q.push_back(Vector3d(-p, 0.0, -p));

        schd.Qwt.push_back(1.0 / 15.0);
        schd.Qwt.push_back(1.0 / 15.0);
        schd.Qwt.push_back(1.0 / 15.0);
        schd.Qwt.push_back(1.0 / 15.0);
        schd.Qwt.push_back(1.0 / 15.0);
        schd.Qwt.push_back(1.0 / 15.0);
        schd.Qwt.push_back(1.0 / 15.0);
        schd.Qwt.push_back(1.0 / 15.0);
        schd.Qwt.push_back(1.0 / 15.0);
        schd.Qwt.push_back(1.0 / 15.0);
        schd.Qwt.push_back(1.0 / 15.0);
        schd.Qwt.push_back(1.0 / 15.0);
      }
      else
      {
        cout << "set pseudo to 4, 6, 12, or 18" << endl;
        exit(0);
      }
    }
    else schd.walkerBasis = ORBITALS;

    //minimal checking for correctness
    //wavefunction
    schd.wavefunctionType = algorithm::to_lower_copy(input.get("wavefunction.name", "jastrowslater"));
    
    //correlatedwavefunction options
    schd.hf = algorithm::to_lower_copy(input.get("wavefunction.hfType", "rhf"));
    schd.ifComplex = input.get("wavefunction.complex", false);
    schd.uagp = input.get("wavefunction.uagp", false); 
    optional< property_tree::iptree& > child = input.get_child_optional("wavefunction.correlators");
    if (child) {
      for (property_tree::iptree::value_type &correlator : input.get_child("wavefunction.correlators")) {
        int siteSize = stoi(correlator.first);
        string file = correlator.second.data();
	    schd.correlatorFiles[siteSize] = file;
      }
    }

    //resonating wave function
    schd.numResonants = input.get("wavefunction.numResonants", 1);
    schd.singleJastrow = input.get("wavefunction.singleJastrow", true);
    schd.readTransOrbs = input.get("wavefunction.readTransOrbs", true);
   
    //noci
    schd.nNociSlater = input.get("wavefunction.nNociSlater", 1); //this is really the same as numResonants

    // permuted wave function
    schd.numPermutations = input.get("wavefunction.numPermutations", 1);

    //ci and lanczos
    schd.nciAct = input.get("wavefunction.numAct", -1);
    schd.overlapCutoff = input.get("wavefunction.overlapCutoff", 1.e-5);
    if (schd.wavefunctionType == "sci") schd.ciCeption = true;
    else schd.ciCeption = false;
    schd.determinantFile = input.get("wavefunction.determinants", ""); //used for both sci and starting det
    schd.alpha = input.get("wavefunction.alpha", 0.01); //lanczos

    //rbm
    schd.numHidden = input.get("wavefunction.numHidden", 1);

    //realspace
    schd.fourBodyJastrow = false;
    optional< property_tree::iptree& > fbj = input.get_child_optional("wavefunction.fourBodyJastrow");
    if (fbj) {
      schd.fourBodyJastrow = true;
      string fbjBasis = input.get("wavefunction.fourBodyJastrow", "NC");
      if (fbjBasis == "NC") schd.fourBodyJastrowBasis = NC;
      else if (fbjBasis == "sNC") schd.fourBodyJastrowBasis = sNC;
      else if (fbjBasis == "AB2") schd.fourBodyJastrowBasis = AB2;
      else if (fbjBasis == "sAB2") schd.fourBodyJastrowBasis = sAB2;
      else if (fbjBasis == "spAB2") schd.fourBodyJastrowBasis = spAB2;
      else if (fbjBasis == "asAB2") {
          schd.fourBodyJastrowBasis = asAB2;
          readActiveSpaceOrbs(schd.asAO);
      }
      else if (fbjBasis == "SS") schd.fourBodyJastrowBasis = SS;
      else if (fbjBasis == "SG") schd.fourBodyJastrowBasis = SG;
      else if (fbjBasis == "G")  {
        schd.fourBodyJastrowBasis = G;
        readGridGaussians(schd.gridGaussians);
      }
    }
    schd.Qmax = input.get("wavefunction.Qmax", 6);
    schd.QmaxEEN = input.get("wavefunction.QmaxEEN", 3);
    schd.noENCusp = input.get("wavefunction.noENCusp", false); //sets 0th order EN jastrow parameter to 0
    schd.noEECusp = input.get("wavefunction.noEECusp", false); //sets 0th order EE jastrow parameter to 0
    schd.addENCusp = input.get("wavefunction.addENCusp", false);  //sets 0th order EN jastrow parameter to -Z
    schd.enforceENCusp = input.get("wavefunction.enforceENCusp", false); //enforces EN cusp condition for slater orbitals, when true noENCusp must be true (otherwise interferes with cusp)
    schd.enforceCusp = input.get("wavefunction.enforceCusp", false); //enforces that three-body jastrows do not interfere with EE, EN cusp condition
    schd.testCusp = input.get("wavefunction.testCusp", false); //tests cusp conditions before dmc calculation
    schd.sigma = input.get("wavefunction.sigma", 1.0);

    //hamiltonian
    string hamString = algorithm::to_lower_copy(input.get("hamiltonian", "abinitio"));
    if (hamString == "abinitio") schd.Hamiltonian = ABINITIO;
    else if (hamString == "hubbard") schd.Hamiltonian = HUBBARD;
   

    //sampling
    schd.epsilon = input.get("sampling.epsilon", 1.e-7);
    schd.screen = input.get("sampling.screentol", 1.e-8);
    schd.ctmc = input.get("sampling.ctmc", true); //if this is false, metropolis is used!
    schd.deterministic = input.get("sampling.deterministic", false);
    schd.stochasticIter = input.get("sampling.stochasticIter", 1e4);
    schd.integralSampleSize = input.get("sampling.integralSampleSize", 10);
    schd.seed = input.get("sampling.seed", getTime());
    schd.sampleEveryRt = input.get("sampling.sampleEveryRt", true);
    
    //gfmc 
    schd.maxIter = input.get("sampling.maxIter", 50); //note: parameter repeated in optimizer for vmc
    schd.nwalk = input.get("sampling.nwalk", 100);
    schd.tau = input.get("sampling.tau", 0.001);
    schd.fn_factor = input.get("sampling.fn_factor", 1.0);
    schd.nGeneration = input.get("sampling.nGeneration", 30.0);
    
    //FCIQMC options
    schd.nAttemptsEach = input.get("sampling.nAttemptsEach", 1);
    schd.mainMemoryFac = input.get("sampling.mainMemoryFac", 5.0);
    schd.spawnMemoryFac = input.get("sampling.spawnMemoryFac", 5.0);
    schd.shiftDamping = input.get("sampling.shiftDamping", 0.01);
    schd.initialShift = input.get("sampling.initialShift", 0.0);
    schd.minSpawn = input.get("sampling.minSpawn", 0.01);
    schd.minPop = input.get("sampling.minPop", 1.0);
    schd.initialPop = input.get("sampling.initialPop", 100.0);
    schd.targetPop = input.get("sampling.targetPop", 1000.0);

    //realspace
    schd.rStepSize = input.get("sampling.rStepSize", 0.1);
    string stepType = input.get("sampling.rStepType", "spherical");
    if (stepType == "spherical") schd.rStepType = SPHERICAL;
    else if (stepType == "simple") schd.rStepType = SIMPLE;
    else if (stepType == "dmc") schd.rStepType = DMC;
    else if (stepType == "gaussian") schd.rStepType = GAUSSIAN;
    schd.doTMove = input.get("sampling.doTMove", false);
    schd.scaledVelocity = input.get("sampling.scaledVelocity", true);

    //trans
    schd.nMaxMacroIter = input.get("sampling.macroIter", 1);
    schd.nMaxMicroIter = input.get("sampling.microIter", 1);
    schd.maxMacroIter = input.get("sampling.maxMacroIter", 1);
 

    //optimization
    string method = algorithm::to_lower_copy(input.get("optimizer.method", "amsgrad")); 
    if (method == "amsgrad") schd.method = amsgrad;
    else if (method == "amsgrad_sgd") schd.method = amsgrad_sgd;
    else if (method == "sgd") schd.method = sgd;
    else if (method == "sr") schd.method = sr;
    else if (method == "lm") {
      schd.method = linearmethod;
      //unfortunately this has to be done here for amsgrad to work correctly
      schd.stepsizes = {0.1, 0.01, 1.0};
      child = input.get_child_optional("optimizer.stepsizes");
      if (child) {
        schd.stepsizes.resize(0);
        for (property_tree::iptree::value_type &step : input.get_child("optimizer.stepsizes")) {
          schd.stepsizes.push_back(stod(step.second.data()));
        }
      }
    }

    //general options
    schd.restart = input.get("optimizer.restart", false);
    schd.fullRestart = input.get("optimizer.fullRestart", false);
    child = input.get_child_optional("sampling.maxIter"); //to ensure maxiter is not reassigned
    if (!child) schd.maxIter = input.get("optimizer.maxIter", 50);
    schd.avgIter = input.get("optimizer.avgIter", 0);
    schd.stepsize = input.get("optimizer.stepsize", 0.001);
    schd.optimizeOrbs = input.get("optimizer.optimizeOrbs", true);
    schd.optimizeCiCoeffs = input.get("optimizer.optimizeCiCoeffs", true);
    schd.optimizeCps = input.get("optimizer.optimizeCps", true);
    schd.optimizeJastrow = input.get("optimizer.optimizeJastrow", true);//this is only used in jrbm, doesn't affect jslater
    schd.optimizeRBM = input.get("optimizer.optimizeRBM", true);
    schd.optimizeBackflow = input.get("optimizer.optimizeBackflow", true);
    
    // amsgrad, sgd
    schd.decay2 = input.get("optimizer.decay2", 0.001);
    schd.decay1 = input.get("optimizer.decay1", 0.1);
    schd.momentum = input.get("optimizer.momentum", 0.);
    
    //lm, sr options
    schd.cgIter = input.get("optimizer.cgIter", 20);
    schd.sDiagShift = input.get("optimizer.sDiagShift", 0.0);
    schd.hDiagShift = input.get("optimizer.hDiagShift", 0.1);
    schd.decay = input.get("optimizer.decay", 0.65);
    schd.sgdIter = input.get("optimizer.sgdIter", 1);
    schd.sgdStepsize = input.get("sgdStepsize", 0.1); 
    schd.CorrSampleFrac = input.get("optimizer.corrSampleFrac", 0.35);
    schd.direct = input.get("optimizer.direct", true);
    schd.dTol = input.get("optimizer.dTol", 1.e-3);
    schd.cgTol = input.get("optimizer.cgTol", 1.e-3);
    schd.tol = input.get("tol", 0.); 
    schd.random = input.get("optimizer.random", false);
    schd.nVec = input.get("optimizer.nVec", 500);
    schd.r = input.get("optimizer.r", 10);
    schd.error = input.get("optimizer.error", 0.01);

    //lm for ci
    schd.diagMethod = input.get("optimizer.diagMethod", "power");
    schd.powerShift = input.get("optimizer.powerShift", 10);


    //debug and print options
    schd.printLevel = input.get("print.level", 0);
    schd.printVars = input.get("print.vars", false);
    schd.printGrad = input.get("print.grad", false);
    schd.printOpt = input.get("print.opt", false);
    schd.debug = input.get("print.debug", false);
    
    //deprecated
    schd.actWidth = input.get("wavefunction.actWidth", 100);
    schd.numActive = input.get("wavefunction.numActive", -1);
    schd.expCorrelator = input.get("wavefunction.expCorrelator", false); 
    schd.PTlambda = input.get("PTlambda", 0.);
    schd.beta = input.get("beta", 1.);
    schd.excitationLevel = input.get("excitationLevel", 10);
    schd.sgdStepsize = input.get("optimizer.sgdStepsize", 0.1); 
    schd.doHessian = input.get("optimizer.doHessian", false);
    
  }

#ifndef SERIAL
  boost::mpi::communicator world;
  mpi::broadcast(world, schd, 0);
#endif
}

void readCorrelator(const std::pair<int, std::string>& p,
		    std::vector<Correlator>& correlators) {
  readCorrelator(p.second, p.first, correlators);
}

void readCorrelator(std::string input, int correlatorSize,
		    std::vector<Correlator>& correlators) {
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
    correlators.push_back(Correlator(asites, bsites));
  }
}


void readHF(MatrixXd& HfmatrixA, MatrixXd& HfmatrixB, std::string hf) 
{
  if (hf == "rhf" || hf == "ghf") {
    ifstream dump("hf.txt");
    for (int i = 0; i < HfmatrixA.rows(); i++) {
      for (int j = 0; j < HfmatrixA.rows(); j++){
        dump >> HfmatrixA(i, j);
	    HfmatrixB(i, j) = HfmatrixA(i, j);
      }
    }
  }
  else {
      ifstream dump("hf.txt");
      for (int i = 0; i < HfmatrixA.rows(); i++)
	{
	  for (int j = 0; j < HfmatrixA.rows(); j++)
	    dump >> HfmatrixA(i, j);
	  for (int j = 0; j < HfmatrixB.rows(); j++)
	    dump >> HfmatrixB(i, j);
	}
    }
/*
  if (schd.optimizeOrbs) {
    //double scale = pow(1.*HfmatrixA.rows(), 0.5);
    double scale = 0.05 * HfmatrixA.maxCoeff();
    HfmatrixA += scale * MatrixXd::Random(HfmatrixA.rows(), HfmatrixA.cols());
    HfmatrixB += scale * MatrixXd::Random(HfmatrixB.rows(), HfmatrixB.cols());
  }
*/
}
void readHF(MatrixXcd& HfmatrixA, MatrixXcd& HfmatrixB, std::string hf) 
{
  if (hf == "rhf" || hf == "ghf") {
    ifstream dump("hf.txt");
    for (int i = 0; i < HfmatrixA.rows(); i++) {
      for (int j = 0; j < HfmatrixA.rows(); j++){
        dump >> HfmatrixA(i, j);
	    HfmatrixB(i, j) = HfmatrixA(i, j);
      }
    }
  }
  else {
      ifstream dump("hf.txt");
      for (int i = 0; i < HfmatrixA.rows(); i++)
	{
	  for (int j = 0; j < HfmatrixA.rows(); j++)
	    dump >> HfmatrixA(i, j);
	  for (int j = 0; j < HfmatrixB.rows(); j++)
	    dump >> HfmatrixB(i, j);
	}
    }
/*
  if (schd.optimizeOrbs) {
    //double scale = pow(1.*HfmatrixA.rows(), 0.5);
    double scale = 0.05 * HfmatrixA.maxCoeff();
    HfmatrixA += scale * MatrixXd::Random(HfmatrixA.rows(), HfmatrixA.cols());
    HfmatrixB += scale * MatrixXd::Random(HfmatrixB.rows(), HfmatrixB.cols());
  }
*/
}

void readGeometry(vector<Vector3d>& Ncoords,
                  vector<double>  & Ncharge,
                  vector<int>  & uniqueAtoms,
                  vector<int>  & uniqueAtomsMap,
                  vector<int> & Nbasis,
                  vector<vector<int>> & NSbasis,
                  vector<vector<int>> & NPbasis,
                  gaussianBasis& gBasis) {
  int N = gBasis.natm;
  Ncoords.resize(N);
  Ncharge.resize(N);
  Nbasis.assign(N, 0);
  NSbasis.resize(N);
  NPbasis.resize(N);

  int stride = gBasis.atm.size()/N;
  for (int i=0; i<N; i++) {
    Ncharge[i] = gBasis.atm[i*stride];
    Ncoords[i][0] = gBasis.env[ gBasis.atm[i*stride+1] +0];
    Ncoords[i][1] = gBasis.env[ gBasis.atm[i*stride+1] +1];
    Ncoords[i][2] = gBasis.env[ gBasis.atm[i*stride+1] +2];
  }

  int orb = 0;
  for (int i = 0; i < gBasis.nbas; i++) {
    int index = gBasis.bas[i * 8];
    int l = gBasis.bas[i * 8 + 1];
    int n = gBasis.bas[i * 8 + 3];
    int numOrbs = n * (2 * l + 1);

    if (gBasis.IntegralType == "cart") {
        if (l == 2) { numOrbs = n * 6; }
        else if (l == 3) { numOrbs = n * 10; }
        else if (l == 4) { numOrbs = n * 15; }
    }

    //total number of orbitals for each atom
    Nbasis[index] += numOrbs;

    //map to s orbitals for each atom
    if (l == 0) {
        for (int j = orb; j < orb + numOrbs; j++) { NSbasis[index].push_back(j); }
    }

    //map to p orbitals for each atom
    if (l == 1) {
        for (int j = orb; j < orb + numOrbs; j++) { NPbasis[index].push_back(j); }
    }

    orb += numOrbs;
  }

  /*
  for (int i = 0; i < NSbasis.size(); i++)
  {
    for (int j = 0; j < NSbasis[i].size(); j++)
    {
      cout << NSbasis[i][j] << endl;
    }
    cout << endl;
  }
  cout << endl;
  for (int i = 0; i < NPbasis.size(); i++)
  {
    for (int j = 0; j < NPbasis[i].size(); j++)
    {
      cout << NPbasis[i][j] << endl;
    }
    cout << endl;
  }
  cout << endl;
  */

  //map for unique atoms
  for (int i = 0; i < Ncharge.size(); i++)
  {
    if (std::find(uniqueAtoms.begin(), uniqueAtoms.end(), Ncharge[i]) == uniqueAtoms.end())
    {
      uniqueAtoms.push_back(Ncharge[i]);
    }
  }
  for (int i = 0; i < Ncharge.size(); i++)
  {
    int index = std::find(uniqueAtoms.begin(), uniqueAtoms.end(), Ncharge[i]) - uniqueAtoms.begin();
    uniqueAtomsMap.push_back(index);
  }

  /*
  for (int i = 0; i < uniqueAtoms.size(); i++)
  {
    cout << uniqueAtoms[i] << " | ";
  }
  cout << endl;
  for (int i = 0; i < uniqueAtomsMap.size(); i++)
  {
    cout << uniqueAtomsMap[i] << " | ";
  }
  cout << endl;
  */

}

//reads in indices for active space atomic orbitals
void readActiveSpaceOrbs(vector<int> &asAO)
{
  ifstream f("asAO.txt");
  while (f.good())
  {
    int index;
    f >> index;
    if (f.eof()) break;
    asAO.push_back(index);
  }
}

void readGridGaussians(vector<pair<double,Vector3d>> &gridGaussians)
{
  ifstream f("gaussians.txt");
  while (f.good())
  {
    pair<double, Vector3d> g;

    f >> g.first;
    f >> g.second(0);
    f >> g.second(1);
    f >> g.second(2);

    gridGaussians.push_back(g);
  }
  gridGaussians.pop_back();
  //for (int i = 0; i < gridGaussians.size(); i++) cout << gridGaussians[i].first << " " << gridGaussians[i].second.transpose() << endl;
}

void readPairMat(MatrixXd& pairMat) 
{
  ifstream dump("pairMat.txt");
  for (int i = 0; i < pairMat.rows(); i++) {
    for (int j = 0; j < pairMat.rows(); j++){
      dump >> pairMat(i, j);
    }
  }
}

void readMat(MatrixXd& mat, std::string fileName) 
{
  ifstream dump(fileName);
  for (int i = 0; i < mat.rows(); i++) {
    for (int j = 0; j < mat.cols(); j++){
      dump >> mat(i, j);
    }
  }
}

void readMat(MatrixXcd& mat, std::string fileName) 
{
  ifstream dump(fileName);
  for (int i = 0; i < mat.rows(); i++) {
    for (int j = 0; j < mat.cols(); j++){
      dump >> mat(i, j);
    }
  }
}

void readDeterminants(std::string input, vector<Determinant> &determinants,
                      vector<double> &ciExpansion)
{
  ifstream dump(input.c_str());
  while (dump.good())
    {
      std::string Line;
      std::getline(dump, Line);

      trim_if(Line, is_any_of(", \t\n"));
      
      vector<string> tok;
      boost::split(tok, Line, is_any_of(", \t\n"), token_compress_on);

      if (tok.size() > 2 )
	{
	  ciExpansion.push_back(atof(tok[0].c_str()));
	  determinants.push_back(Determinant());
	  Determinant& det = *determinants.rbegin();
	  for (int i=0; i<Determinant::norbs; i++) 
	    {
	      if (boost::iequals(tok[1+i], "2")) 
		{
		  det.setoccA(i, true);
		  det.setoccB(i, true);
		}
	      else if (boost::iequals(tok[1+i], "a")) 
		{
		  det.setoccA(i, true);
		  det.setoccB(i, false);
		}
	      if (boost::iequals(tok[1+i], "b")) 
		{
		  det.setoccA(i, false);
		  det.setoccB(i, true);
		}
	      if (boost::iequals(tok[1+i], "0")) 
		{
		  det.setoccA(i, false);
		  det.setoccB(i, false);
		}
	    }

	  //***************I AM USING alpha-beta format here, but the wavefunction is coming from Dice that uses alpha0 beta0 alpha1 beta1... format
	  //So the signs need to be adjusted appropriately
	  //cout << det<<"   "<<getParityForDiceToAlphaBeta(det)<<endl;
	  *ciExpansion.rbegin() *= getParityForDiceToAlphaBeta(det);
	}
    }
}

void readDeterminants(std::string input, std::vector<int>& ref, std::vector<int>& open, std::vector<std::array<VectorXi, 2>>& ciExcitations,
        std::vector<int>& ciParity, std::vector<double>& ciCoeffs)
{
  int norbs = Determinant::norbs;
  ifstream dump(input.c_str());
  bool isFirst = true;
  Determinant refDet;
  VectorXi sizes = VectorXi::Zero(10);
  int numDets = 0;
  
  while (dump.good()) {
    std::string Line;
    std::getline(dump, Line);

    boost::trim_if(Line, boost::is_any_of(", \t\n"));
    
    vector<string> tok;
    boost::split(tok, Line, boost::is_any_of(", \t\n"), boost::token_compress_on);


    if (tok.size() > 2 ) {
      if (isFirst) {//first det is ref
        isFirst = false;
        ciCoeffs.push_back(atof(tok[0].c_str()));
        ciParity.push_back(1);
        std::array<VectorXi, 2> empty;
        ciExcitations.push_back(empty);
        vector<int> closedBeta, openBeta; //no ghf det structure, so artificially using vector of ints, alpha followed by beta
        for (int i=0; i<norbs; i++) {
          if (boost::iequals(tok[1+i], "2")) {
            refDet.setoccA(i, true);
            refDet.setoccB(i, true);
            ref.push_back(i);
            closedBeta.push_back(i + norbs);
          }
          else if (boost::iequals(tok[1+i], "a")) {
            refDet.setoccA(i, true);
            ref.push_back(i);
            openBeta.push_back(i + norbs);
          }
          else if (boost::iequals(tok[1+i], "b")) {
            refDet.setoccB(i, true);
            closedBeta.push_back(i + norbs);
            open.push_back(i);
          }
          else if (boost::iequals(tok[1+i], "0")) {
            open.push_back(i);
            openBeta.push_back(i + norbs);
          }
        }
        ref.insert(ref.end(), closedBeta.begin(), closedBeta.end());
        open.insert(open.end(), openBeta.begin(), openBeta.end());
      }
      else {
        vector<int> desA, creA, desB, creB;
        for (int i=0; i<norbs; i++) {
          if (boost::iequals(tok[1+i], "2")) {
            if (!refDet.getoccA(i)) creA.push_back(i);
            if (!refDet.getoccB(i)) creB.push_back(i);
          }
          else if (boost::iequals(tok[1+i], "a")) {
            if (!refDet.getoccA(i)) creA.push_back(i);
            if (refDet.getoccB(i)) desB.push_back(i);
          }
          else if (boost::iequals(tok[1+i], "b")) {
            if (refDet.getoccA(i)) desA.push_back(i);
            if (!refDet.getoccB(i)) creB.push_back(i);
          }
          else if (boost::iequals(tok[1+i], "0")) {
            if (refDet.getoccA(i)) desA.push_back(i);
            if (refDet.getoccB(i)) desB.push_back(i);
          }
        }
        VectorXi cre = VectorXi::Zero(creA.size() + creB.size());
        VectorXi des = VectorXi::Zero(desA.size() + desB.size());
        for (int i = 0; i < creA.size(); i++) {
          des[i] = std::search_n(ref.begin(), ref.end(), 1, desA[i]) - ref.begin();
          cre[i] = std::search_n(open.begin(), open.end(), 1, creA[i]) - open.begin();
          //cre[i] = creA[i];
        }
        for (int i = 0; i < creB.size(); i++) {
          des[i + desA.size()] = std::search_n(ref.begin(), ref.end(), 1, desB[i] + norbs) - ref.begin();
          cre[i + creA.size()] = std::search_n(open.begin(), open.end(), 1, creB[i] + norbs) - open.begin();
          //cre[i + creA.size()] = creB[i] + norbs;
        }
        std::array<VectorXi, 2> excitations;
        excitations[0] = des;
        excitations[1] = cre;
        if (cre.size() > schd.excitationLevel) continue;
        numDets++;
        ciCoeffs.push_back(atof(tok[0].c_str()));
        ciParity.push_back(refDet.parityA(creA, desA) * refDet.parityB(creB, desB));
        ciExcitations.push_back(excitations);
        if (cre.size() < 10) sizes(cre.size())++;
      }
    }
  }
  if (commrank == 0) {
    cout << "Rankwise number of excitations " << sizes.transpose() << endl;
    cout << "Number of determinants " << numDets << endl << endl;
  }
}

void readDeterminantsGHF(std::string input, std::vector<int>& ref, std::vector<int>& open, std::vector<std::array<VectorXi, 2>>& ciExcitations,
        std::vector<int>& ciParity, std::vector<double>& ciCoeffs)
{
  int norbs = Determinant::norbs;
  ifstream dump(input.c_str());
  bool isFirst = true;
  Determinant refDet;

  while (dump.good()) {
    std::string Line;
    std::getline(dump, Line);

    boost::trim_if(Line, boost::is_any_of(", \t\n"));
    
    vector<string> tok;
    boost::split(tok, Line, boost::is_any_of(", \t\n"), boost::token_compress_on);

    if (tok.size() > 2 ) {
      if (isFirst) {//first det is ref
        isFirst = false;
        ciCoeffs.push_back(atof(tok[0].c_str()));
        ciParity.push_back(1);
        std::array<VectorXi, 2> empty;
        ciExcitations.push_back(empty);
        for (int i=0; i<norbs; i++) {
          if (boost::iequals(tok[1+i], "2")) {
            refDet.setoccA(2*i, true);
            refDet.setoccA(2*i+1, true);
            ref.push_back(2*i);
            ref.push_back(2*i+1);
          }
          else if (boost::iequals(tok[1+i], "a")) {
            refDet.setoccA(2*i, true);
            ref.push_back(2*i);
            open.push_back(2*i+1);
          }
          else if (boost::iequals(tok[1+i], "b")) {
            refDet.setoccA(2*i+1, true);
            ref.push_back(2*i+1);
            open.push_back(2*i);
          }
          else if (boost::iequals(tok[1+i], "0")) {
            open.push_back(2*i);
            open.push_back(2*i+1);
          }
        }
      }
      else {
        ciCoeffs.push_back(atof(tok[0].c_str()));
        vector<int> des, cre;
        for (int i=0; i<norbs; i++) {
          if (boost::iequals(tok[1+i], "2")) {
            if (!refDet.getoccA(2*i)) cre.push_back(2*i);
            if (!refDet.getoccA(2*i+1)) cre.push_back(2*i+1);
          }
          else if (boost::iequals(tok[1+i], "a")) {
            if (!refDet.getoccA(2*i)) cre.push_back(2*i);
            if (refDet.getoccA(2*i+1)) des.push_back(2*i+1);
          }
          else if (boost::iequals(tok[1+i], "b")) {
            if (refDet.getoccA(2*i)) des.push_back(2*i);
            if (!refDet.getoccA(2*i+1)) cre.push_back(2*i+1);
          }
          else if (boost::iequals(tok[1+i], "0")) {
            if (refDet.getoccA(2*i)) des.push_back(2*i);
            if (refDet.getoccA(2*i+1)) des.push_back(2*i+1);
          }
        }
        ciParity.push_back(refDet.parityA(cre, des));
        VectorXi creV = VectorXi::Zero(cre.size());
        VectorXi desV = VectorXi::Zero(des.size());
        for (int i = 0; i < cre.size(); i++) {
          desV[i] = std::search_n(ref.begin(), ref.end(), 1, des[i]) - ref.begin();
          creV[i] = std::search_n(open.begin(), open.end(), 1, cre[i]) - open.begin();
          //creV[i] = cre[i];
        }
        std::array<VectorXi, 2> excitations;
        excitations[0] = desV;
        excitations[1] = creV;
        ciExcitations.push_back(excitations);
      }
    }
  }
}


void readDeterminants(std::string input, std::array<std::vector<int>, 2>& ref, std::array<std::vector<std::array<Eigen::VectorXi, 2>>, 2>& ciExcitations,
        std::vector<double>& ciParity, std::vector<double>& ciCoeffs)
{
  int norbs = Determinant::norbs;
  int nact = norbs;
  if (schd.nciAct > 0) nact = schd.nciAct;
  int ncore = 0;
  if (schd.nciCore > 0) ncore = schd.nciCore;
  ifstream dump(input.c_str());
  bool isFirst = true;
  Determinant refDet;
  for (int i = 0; i < ncore; i++) {
    refDet.setoccA(i, true);
    refDet.setoccB(i, true);
  }
  VectorXi sizes = VectorXi::Zero(10);
  int numDets = 0;
  
  while (dump.good()) {
    std::string Line;
    std::getline(dump, Line);

    boost::trim_if(Line, boost::is_any_of(", \t\n"));
    
    vector<string> tok;
    boost::split(tok, Line, boost::is_any_of(", \t\n"), boost::token_compress_on);

    if (tok.size() > 2 ) {
      if (isFirst) {// first det is ref
        isFirst = false;
        ciCoeffs.push_back(atof(tok[0].c_str()));
        ciParity.push_back(1);
        std::array<VectorXi, 2> empty;
        ciExcitations[0].push_back(empty);
        ciExcitations[1].push_back(empty);
        vector<int> closedBeta, openBeta; 
        for (int i = 0; i < nact; i++) {
          if (boost::iequals(tok[1+i], "2")) {
            refDet.setoccA(ncore + i, true);
            refDet.setoccB(ncore + i, true);
            ref[0].push_back(ncore + i);
            ref[1].push_back(ncore + i);
          }
          else if (boost::iequals(tok[1+i], "a")) {
            refDet.setoccA(ncore + i, true);
            ref[0].push_back(ncore + i);
          }
          else if (boost::iequals(tok[1+i], "b")) {
            refDet.setoccB(ncore + i, true);
            ref[1].push_back(ncore + i);
          }
        }
        numDets++;
        sizes(0) = 1;
      }
      else {
        vector<int> desA, creA, desB, creB;
        for (int i = 0; i < nact; i++) {
          if (boost::iequals(tok[1+i], "2")) {
            if (!refDet.getoccA(ncore + i)) creA.push_back(ncore + i);
            if (!refDet.getoccB(ncore + i)) creB.push_back(ncore + i);
          }
          else if (boost::iequals(tok[1+i], "a")) {
            if (!refDet.getoccA(ncore + i)) creA.push_back(ncore + i);
            if (refDet.getoccB(ncore + i)) desB.push_back(ncore + i);
          }
          else if (boost::iequals(tok[1+i], "b")) {
            if (refDet.getoccA(ncore + i)) desA.push_back(ncore + i);
            if (!refDet.getoccB(ncore + i)) creB.push_back(ncore + i);
          }
          else if (boost::iequals(tok[1+i], "0")) {
            if (refDet.getoccA(ncore + i)) desA.push_back(ncore + i);
            if (refDet.getoccB(ncore + i)) desB.push_back(ncore + i);
          }
        }

        std::array<VectorXi, 2> excitationsA;
        excitationsA[0] = VectorXi::Zero(creA.size());
        excitationsA[1] = VectorXi::Zero(desA.size());
        for (int i = 0; i < creA.size(); i++) {
          //des[i] = std::search_n(ref.begin(), ref.end(), 1, desA[i]) - ref.begin();
          //cre[i] = std::search_n(open.begin(), open.end(), 1, creA[i]) - open.begin();
          excitationsA[0](i) = desA[i];
          excitationsA[1](i) = creA[i];
        }

        std::array<VectorXi, 2> excitationsB;
        excitationsB[0] = VectorXi::Zero(creB.size());
        excitationsB[1] = VectorXi::Zero(desB.size());
        for (int i = 0; i < creB.size(); i++) {
          //des[i + desA.size()] = std::search_n(ref.begin(), ref.end(), 1, desB[i] + norbs) - ref.begin();
          //cre[i + creA.size()] = std::search_n(open.begin(), open.end(), 1, creB[i] + norbs) - open.begin();
          excitationsB[0](i) = desB[i];
          excitationsB[1](i) = creB[i];
        }
        
        if (creA.size() + creB.size() > schd.excitationLevel) continue;
        numDets++;
        ciCoeffs.push_back(atof(tok[0].c_str()));
        ciParity.push_back(refDet.parityA(creA, desA) * refDet.parityB(creB, desB));
        ciExcitations[0].push_back(excitationsA);
        ciExcitations[1].push_back(excitationsB);
        if (creA.size() + creB.size() < 10) sizes(creA.size() + creB.size())++;
      }
    }
  }
  //if (commrank == 0) {
  //  cout << "Rankwise number of excitations " << sizes.transpose() << endl;
  //  cout << "Number of determinants " << numDets << endl << endl;
  //}
}

// same as above but for binary files
void readDeterminantsBinary(std::string input, std::array<std::vector<int>, 2>& ref, std::array<std::vector<std::array<Eigen::VectorXi, 2>>, 2>& ciExcitations,
        std::vector<double>& ciParity, std::vector<double>& ciCoeffs)
{
  int norbs = Determinant::norbs;
  int nact = norbs;
  if (schd.nciAct > 0) nact = schd.nciAct;
  int ncore = 0;
  if (schd.nciCore > 0) ncore = schd.nciCore;
  int ndetsDice = 0, norbsDice = 0;
  ifstream dump(input, ios::binary);
  dump.read((char*) &ndetsDice, sizeof(int));
  dump.read((char*) &norbsDice, sizeof(int));
  bool isFirst = true;
  Determinant refDet;
  for (int i = 0; i < ncore; i++) {
    refDet.setoccA(i, true);
    refDet.setoccB(i, true);
  }
  VectorXi sizes = VectorXi::Zero(10);
  int numDets = 0;
  
  for (int n = 0; n < ndetsDice; n++) {
    if (isFirst) {// first det is ref
      isFirst = false;
      double ciCoeff;
      dump.read((char*) &ciCoeff, sizeof(double));
      ciCoeffs.push_back(ciCoeff);
      ciParity.push_back(1);
      std::array<VectorXi, 2> empty;
      ciExcitations[0].push_back(empty);
      ciExcitations[1].push_back(empty);
      vector<int> closedBeta, openBeta; 
      for (int i = ncore; i < ncore + norbsDice; i++) {
        char detocc;
        dump.read((char*) &detocc, sizeof(char));
        if (detocc == '2') {
          refDet.setoccA(i, true);
          refDet.setoccB(i, true);
          ref[0].push_back(i);
          ref[1].push_back(i);
        }
        else if (detocc == 'a') {
          refDet.setoccA(i, true);
          ref[0].push_back(i);
        }
        else if (detocc == 'b') {
          refDet.setoccB(i, true);
          ref[1].push_back(i);
        }
      }
      numDets++;
      sizes(0) = 1;
    }
    else {
      double ciCoeff;
      dump.read((char*) &ciCoeff, sizeof(double));
      vector<int> desA, creA, desB, creB;
      for (int i = ncore; i < ncore + norbsDice; i++) {
        char detocc;
        dump.read((char*) &detocc, sizeof(char));
        if (detocc == '2') {
          if (!refDet.getoccA(i)) creA.push_back(i);
          if (!refDet.getoccB(i)) creB.push_back(i);
        }
        else if (detocc == 'a') {
          if (!refDet.getoccA(i)) creA.push_back(i);
          if (refDet.getoccB(i)) desB.push_back(i);
        }
        else if (detocc == 'b') {
          if (refDet.getoccA(i)) desA.push_back(i);
          if (!refDet.getoccB(i)) creB.push_back(i);
        }
        else if (detocc == '0') {
          if (refDet.getoccA(i)) desA.push_back(i);
          if (refDet.getoccB(i)) desB.push_back(i);
        }
      }

      std::array<VectorXi, 2> excitationsA;
      excitationsA[0] = VectorXi::Zero(creA.size());
      excitationsA[1] = VectorXi::Zero(desA.size());
      for (int i = 0; i < creA.size(); i++) {
        //des[i] = std::search_n(ref.begin(), ref.end(), 1, desA[i]) - ref.begin();
        //cre[i] = std::search_n(open.begin(), open.end(), 1, creA[i]) - open.begin();
        excitationsA[0](i) = desA[i];
        excitationsA[1](i) = creA[i];
      }

      std::array<VectorXi, 2> excitationsB;
      excitationsB[0] = VectorXi::Zero(creB.size());
      excitationsB[1] = VectorXi::Zero(desB.size());
      for (int i = 0; i < creB.size(); i++) {
        //des[i + desA.size()] = std::search_n(ref.begin(), ref.end(), 1, desB[i] + norbs) - ref.begin();
        //cre[i + creA.size()] = std::search_n(open.begin(), open.end(), 1, creB[i] + norbs) - open.begin();
        excitationsB[0](i) = desB[i];
        excitationsB[1](i) = creB[i];
      }
      
      if (creA.size() + creB.size() > schd.excitationLevel) continue;
      numDets++;
      ciCoeffs.push_back(ciCoeff);
      ciParity.push_back(refDet.parityA(creA, desA) * refDet.parityB(creB, desB));
      ciExcitations[0].push_back(excitationsA);
      ciExcitations[1].push_back(excitationsB);
      if (creA.size() + creB.size() < 10) sizes(creA.size() + creB.size())++;
    }
  }
  if (commrank == 0) {
    cout << "Rankwise number of excitations " << sizes.transpose() << endl;
    cout << "Number of determinants " << numDets << endl << endl;
  }
}
