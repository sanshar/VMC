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
#ifndef PROPAGATE_HEADER_H
#define PROPAGATE_HEADER_H
#include <Eigen/Dense>
#include <boost/serialization/serialization.hpp>
#include "iowrapper.h"
#include "global.h"
#include "statistics.h"
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#include <list>
#include <utility>
#include <vector>

#ifndef SERIAL
#include "mpi.h"
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif
#include "workingArray.h"

using namespace Eigen;
using namespace boost;
using namespace std;


//generate list of walkers and weights via orbital space VMC
template<typename Walker, typename Wfn>
void generateWalkers(list<pair<Walker, double> >& Walkers, Wfn& w, workingArray& work) {

  auto random = std::bind(std::uniform_real_distribution<double>(0, 1),
                          std::ref(generator));
  
  Walker walk = Walkers.begin()->first;
  
  double ham, ovlp;

  int iter = 0;

  //select a new walker every 30 iterations
  int numSample = Walkers.size();
  int niter = 30*numSample+1;
  auto it = Walkers.begin();
  while(iter < niter) {
    w.HamAndOvlp(walk, ovlp, ham, work);

    double cumovlpRatio = 0;
    //when using uniform probability 1./numConnection * max(1, pi/pj)
    for (int i = 0; i < work.nExcitations; i++)
    {
      cumovlpRatio += abs(work.ovlpRatio[i]);
      //cumovlpRatio += min(1.0, pow(ovlpRatio[i], 2));
      work.ovlpRatio[i] = cumovlpRatio;
    }

    if (iter % 30 == 0) {
      it->first = walk; it->second = 1.0/cumovlpRatio;
      it++;
      if (it == Walkers.end()) break;
    }

    
    double nextDetRandom = random()*cumovlpRatio;
    int nextDet = std::lower_bound(work.ovlpRatio.begin(), 
				   (work.ovlpRatio.begin()+work.nExcitations),
                                   nextDetRandom) - work.ovlpRatio.begin();

    walk.updateWalker(w.getRef(), w.getCorr(), work.excitation1[nextDet], work.excitation2[nextDet]);
  }
};


//generate list of walkers and weights via real space VMC
template<typename Wfn, typename Walker>
void generaterWalkers(std::list<std::pair<Walker, double>> &Walkers, Wfn &w)
{
  auto random = std::bind(std::uniform_real_distribution<double>(0, 1), std::ref(generator));

  Walker walk = Walkers.begin()->first;
  int nelec = walk.d.nelec;

  auto it = Walkers.begin();
  int nIter = 30 * Walkers.size() + 1;
  for (int iter = 0; iter < nIter; iter++)
  {
    for (int elec = 0; elec < nelec; elec++)
    {
        double ovlpProb, proposalProb;
        Vector3d step;
        walk.getStep(step, elec, schd.rStepSize, w.getRef(), w.getCorr(), ovlpProb, proposalProb);
        step += walk.d.coord[elec];
        //if move is simple or gaussian
        if (ovlpProb < -0.5) ovlpProb = std::pow(w.getOverlapFactor(elec, step, walk), 2); 

        //accept or reject move
        if (ovlpProb * proposalProb > random()) walk.updateWalker(elec, step, w.getRef(), w.getCorr());
    }

    if (iter % 30 == 0)
    {
      it->first = walk;
      it->second = 1.0;
      it++;
      if (it == Walkers.end()) break; 
    }
  }
}


/*Take the walker and propagate it for time tau using orbital space continous time algorithm
 */
template<typename Wfn, typename Walker>
  void applyPropogatorContinousTime(Wfn &w, Walker& walk, double& wt, 
				    double& tau, workingArray &work, double& Eshift,
				    double& ham, double& ovlp, double fn_factor) {

  auto random = std::bind(std::uniform_real_distribution<double>(0, 1),
                          std::ref(generator));
  

  double t = tau;
  vector<double> cumHijElements;

  while (t > 0) {
    double Ewalk = walk.d.Energy(I1, I2, coreE);

    w.HamAndOvlp(walk, ovlp, ham, work);

    if (cumHijElements.size()< work.nExcitations) cumHijElements.resize(work.nExcitations);

    double cumHij = 0;

    for (int i = 0; i < work.nExcitations; i++)
    {
      //if sy,x is > 0 then add include contribution to the diagonal
      if (work.HijElement[i]*work.ovlpRatio[i] > 0.0) {
	Ewalk += work.HijElement[i]*work.ovlpRatio[i] * (1+fn_factor);
	cumHij += fn_factor*abs(work.HijElement[i]*work.ovlpRatio[i]); 
	cumHijElements[i] = cumHij; 
      }
      else {
	cumHij += abs(work.HijElement[i]*work.ovlpRatio[i]); 
	cumHijElements[i] = cumHij;
      }
    }

    //ham = Ewalk-cumHij;
    double tsample = -log(random())/cumHij;
    double deltaT = min(t, tsample);
    t -= tsample;

    wt = wt * exp(deltaT*(Eshift - (Ewalk-cumHij) ));

    if (t > 0.0)   {
      double nextDetRandom = random()*cumHij;
      int nextDet = std::lower_bound(cumHijElements.begin(), 
				     (cumHijElements.begin() + work.nExcitations),
				     nextDetRandom) - cumHijElements.begin();    
      walk.updateWalker(w.getRef(), w.getCorr(), work.excitation1[nextDet], work.excitation2[nextDet]);
    }
    
  }
};


/*Take the walker and propagate it for time tau using real space metropolis algorithm
 */
template<typename Wfn, typename Walker>
double applyPropogatorMetropolis(Wfn &w, Walker &walk, double &wt, double tau, double Eshift, double &Eloc)
{
  auto random = std::bind(std::uniform_real_distribution<double>(0, 1), std::ref(generator));

  //double oldEloc = w.rHam(walk);

  //propogate every electron once via metropolis-hastings with dmc propogator
  double acceptedProb = 0.0;
  double dr2_accepted = 0.0, dr2_proposed = 0.0;
  int nelec = walk.d.nelec;
  for (int i = 0; i < nelec; i++)
  {
    double ovlpRatio = 0.0, proposalProb = 0.0;
    Vector3d step;
    walk.doDMCMove(step, i, tau, w.getRef(), w.getCorr(), ovlpRatio, proposalProb);
    dr2_proposed += step.squaredNorm();
    Vector3d newCoord = step + walk.d.coord[i];

    //accept move?
    double ovlpProb = ovlpRatio * ovlpRatio;
    if (ovlpProb * proposalProb > random() && ovlpRatio > 0.0) //accept move based on metropolis criterion and if node is not crossed
    {
      acceptedProb++;
      dr2_accepted += step.squaredNorm();
      walk.updateWalker(i, newCoord, w.getRef(), w.getCorr());
    }
  }
  if (acceptedProb != 0) { dr2_accepted /= acceptedProb; } //don't divide by zero
  dr2_proposed /= double(nelec);
  acceptedProb /= double(nelec);

  //calculate energy and t moves
  double T, Vij, ViI, Vnl, VIJ;
  std::vector<std::vector<double>> viq;
  std::vector<std::vector<Vector3d>> riq;
  Eloc = w.rHam(walk, T, Vij, ViI, Vnl, VIJ, viq, riq);

  double tau_eff = tau * dr2_accepted / dr2_proposed; //effective time step due to metropolis algorithm
  wt *= std::exp(- tau_eff * (Eloc - Eshift));

  //print if walker has weird wt or Eloc
  if (wt > 3.0 || std::isnan(Eloc) || std::isnan(wt))
  {
    cout << "Process: " << commrank << endl;
    cout << "Large weight: " << wt << endl;
    cout << "exponential: " << - tau_eff * (Eloc - Eshift) << endl;
    cout << "effective time step: " << tau_eff << endl;
    cout << "Energy shift: " << Eshift << endl;
    cout << "Local energy: " << Eloc << " | " << w.rHam(walk) << endl;
    cout << "T: " << T << " Vij: " << Vij << " ViI: " << ViI << " Vnl: " << Vnl << " VIJ: " << VIJ << endl;
    cout << "coords" << endl;
    cout << walk.d << endl;
    cout << "RiI" << endl;
    cout << walk.RiN << endl;
    cout << "Rij" << endl;
    cout << walk.Rij << endl;
  }

  //size consistent t-moves
  if (schd.doTMove)
  {
    for (int i = 0; i < nelec; i++)
    {
      //if no moves, continue
      if (viq[i].empty()) { continue; }

      //make heat bath move
      double cumT = 0.0;
      std::vector<double> t;
      //first possible move is no move, with txx = 1.0
      cumT += 1.0;
      t.push_back(cumT);
      for (int q = 0; q < viq[i].size(); q++)
      {
        double txpx = - tau * viq[i][q];
        //cout << txpx << " | " << riq[i][q].transpose() << endl;

        cumT += txpx;
        t.push_back(cumT);
      }
      //get index of move, remember first index is no move
      double move = random() * cumT;
      int index = std::lower_bound(t.begin(), t.end(), move) - t.begin() - 1;
      if (index > -1) { walk.updateWalker(i, riq[i][index], w.getRef(), w.getCorr()); }
      //double ovlpRatio = w.getOverlapFactor(i, riq[i][index], walk);
      //if (ovlpRatio > 0.0) { walk.updateWalker(i, riq[i][index], w.getRef(), w.getCorr()); } //don't cross node
    }
  }

  return acceptedProb;
}


/*
collects together walkers of low weight and splits walkers of large weight to enforce numerical stability

function is used in real space and orbital space code
*/
template<typename Walker>
void reconfigure(std::list<pair<Walker, double> >& walkers) {

  auto random = std::bind(std::uniform_real_distribution<double>(0, 1), std::ref(generator));

  vector< typename std::list<pair<Walker, double> >::iterator > smallWts; smallWts.reserve(10);
  double totalSmallWts = 0.0;

  auto it = walkers.begin();


  //if a walker has wt > 2.0, then ducplicate it int(wt) times
  //accumulate walkers with wt < 0.5, 
  //then combine them and assing it to one of the walkers
  //with a prob. proportional to its wt.
  while (it != walkers.end()) {
    double wt = it->second;

    if (wt < 0.5) {
      totalSmallWts += wt;
      smallWts.push_back(it);
      it++;
    }
    else if (wt > 2.0) {
      int numReplicate = int(wt);
      it->second = it->second/(1.0*numReplicate);
      for (int i=0; i<numReplicate-1; i++) 
	    walkers.push_front(*it); //add at front
      it++;
    }
    else
      it++;


    if (totalSmallWts > 1.0 || it == walkers.end()) {
      double select = random()*totalSmallWts;

      //remove all wts except the selected one
      for (int i=0; i<smallWts.size(); i++) {
	    double bkpwt = smallWts[i]->second;
	    if (select > 0 && select < smallWts[i]->second)
	      smallWts[i]->second = totalSmallWts;
	    else 
	      walkers.erase(smallWts[i]);
	      select -= bkpwt;
      }
      totalSmallWts = 0.0;
      smallWts.resize(0);
    }

  }

  //now redistribute the walkers so that all procs havve roughly the same number
  vector<int> nwalkersPerProc(commsize);
  nwalkersPerProc[commrank] = walkers.size();
  int nTotalWalkers = nwalkersPerProc[commrank];

#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &nTotalWalkers, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &nwalkersPerProc[0], commsize, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif  
  vector<int> procByNWalkers(commsize);
  std::iota(procByNWalkers.begin(), procByNWalkers.end(), 0);
  std::sort(procByNWalkers.begin(), procByNWalkers.end(), [&nwalkersPerProc](size_t i1, size_t i2) {return nwalkersPerProc[i1] < nwalkersPerProc[i2];});

  return;  
};


//continously apply the continous time GFMC algorithm
template<typename Wfn, typename Walker>
  void doGFMCCT(Wfn &w, Walker& walk, double Eshift)
{
  startofCalc = getTime();
  auto random = std::bind(std::uniform_real_distribution<double>(0, 1),
			  std::ref(generator));

  workingArray work;

  //we want to have a set of schd.nwalk walkers and weights
  //we just replciate the input walker nwalk times
  list<pair<Walker, double> > Walkers;
  for (int i=0; i<schd.nwalk; i++)
    Walkers.push_back(std::pair<Walker, double>(walk, 1.0));

  //now actually sample the wavefunction w to generate these walkers
  generateWalkers(Walkers, w, work);

  //normalize so the cumulative weight is schd.nwalk whichi s also the target wt
  double oldwt = .0;
  for (auto it=Walkers.begin(); it!=Walkers.end(); it++) oldwt += it->second;
  for (auto it=Walkers.begin(); it!=Walkers.end(); it++) {
    it->second = it->second*schd.nwalk/oldwt;
  }
  oldwt = schd.nwalk * commsize;
  double targetwt = schd.nwalk;


  //run calculation for niter itarions
  int niter = schd.maxIter, iter = 0;

  //each iteration consists of a time tau 
  //(after every iteration pop control and reconfiguration is performed
  double tau = schd.tau;

  //after every nGeneration iterations, calculate/print the energy
  //the Enum and Eden are restarted after each geneation
  int nGeneration = schd.nGeneration;

  //exponentially decaying average and regular average
  double EavgExpDecay = 0.0, Eavg;
  
  double Enum = 0.0, Eden = 0.0;
  double popControl = 1.0;
  double iterPop = 0.0, olditerPop = oldwt;

  while (iter < niter)
  {
    
    //propagate all walkers for time tau
    iterPop = 0.0;
    for (auto it = Walkers.begin(); it != Walkers.end(); it++) {
      Walker& walk = it->first;
      double& wt = it->second;
      
      double ham=0., ovlp=0.;
      double wtsold = wt;
      applyPropogatorContinousTime(w, walk, wt, schd.tau,
				   work, Eshift, ham, ovlp, 
				   schd.fn_factor);
      
      iterPop += abs(wt);
      Enum += popControl*wt*ham;
      Eden += popControl*wt;	
    }

    //reconfigure
    reconfigure(Walkers);

    //accumulate the total weight across processors
#ifndef SERIAL
    MPI_Allreduce(MPI_IN_PLACE, &iterPop, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

    //all procs have the same Eshift
    Eshift = Eshift - 0.1/tau * log(iterPop/olditerPop);

    //population control and keep a exponentially decaying population
    //control factor
    //double factor = pow(olditerPop/iterPop, 0.1);
    //popControl = pow(popControl, 0.99)*factor;

    olditerPop = iterPop;
    
    if (iter % nGeneration == 0 && iter != 0) {
#ifndef SERIAL
      MPI_Allreduce(MPI_IN_PLACE, &Enum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &Eden, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

      //calculate average energy acorss all procs
      double Ecurrentavg = Enum/Eden;

      //this is the exponentially moving average
      if (iter == nGeneration)
	EavgExpDecay = Ecurrentavg;
      else
	EavgExpDecay = (0.9)*EavgExpDecay +  0.1*Ecurrentavg;


      //this is the average energy, but only start
      //taking averages after 4 generations because 
      //we assume that it takes atleast 4 generations to reach equilibration
      if (iter/nGeneration == 4) 
	Eavg = Ecurrentavg;
      else if (iter/nGeneration > 4) {
	int oldIter = iter/nGeneration;
	Eavg = ((oldIter - 4)*Eavg +  Ecurrentavg)/(oldIter-3);
      }
      else 
	Eavg = EavgExpDecay;

      //growth estimator
      double Egr = Eshift ;
      
      if (commrank == 0)	  
	cout << format("%8i %14.8f  %14.8f  %14.8f  %10i  %8.2f   %14.8f   %8.2f\n") % iter % Ecurrentavg % (EavgExpDecay) %Eavg % (Walkers.size()) % (iterPop/commsize) % Egr % (getTime()-startofCalc);
      
      //restart the Enum and Eden
      Enum=0.0; Eden =0.0;
      popControl = 1.0;
    }
    iter++;
  }
};

//performs the diffusion monte carlo algorithm
template<typename Wfn, typename Walker>
void doDMC(Wfn &w, Walker &walk, double Eshift)
{
  auto random = std::bind(std::uniform_real_distribution<double>(0, 1), std::ref(generator));

  //list of schd.nwalk walkers and weights
  //we replciate the input walker nwalk times
  std::list<std::pair<Walker, double>> Walkers;
  for (int i = 0; i < schd.nwalk; i++) Walkers.push_back(std::pair<Walker, double>(walk, 1.0));

  //sample the wavefunction w to generate walkers
  generaterWalkers(Walkers, w);

  //total weight equals to schd.nwalk;
  double oldwt = schd.nwalk * commsize;
  double targetwt = schd.nwalk * commsize;

  //energy of inital population
  double nE0 = 0.0;
  double dE0 = 0.0;  
  for (auto it = Walkers.begin(); it != Walkers.end(); it++)
  {
      double Eloc = w.rHam(it->first);
      double wt = it->second;
      nE0 += Eloc * wt;
      dE0 += wt;
  }
#ifndef SERIAL
      MPI_Allreduce(MPI_IN_PLACE, &nE0, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &dE0, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
  if (commrank == 0) { cout << "Energy of initial population: " << nE0 / dE0 << endl; }

  //run calculation for niter iterations
  int niter = schd.maxIter;

  //time step
  double tau = schd.tau;

  //after every nGeneration iterations, calculate/print the energy
  int nGeneration = schd.nGeneration;

  //energy average
  double Eavg = 0.0, Eexp = 0.0;
  double Enum = 0.0, Eden = 0.0;
  double E2num = 0.0;
  //error averages
  Statistics Stats;

  double olditerPop = oldwt;
  for (int iter = 0; iter < niter; iter++)
  {
    //propogate walkers time tau and calculate energy
    double iterPop = 0.0;
    double acceptedMoves = 0.0;
    for (auto it = Walkers.begin(); it != Walkers.end(); it++)
    {
      Walker &walk = it->first;
      double &wt = it->second;

      double Eloc = 0.0;
      acceptedMoves += applyPropogatorMetropolis(w, walk, wt, tau, Eshift, Eloc);

      iterPop += wt;
      Enum += wt * Eloc;
      E2num += wt * Eloc * Eloc;
      Stats.push_back(Eloc, wt);
      Eden += wt;
    }

    double totalMoves = double(Walkers.size());
    //accumulate the total weight across processors
#ifndef SERIAL
    MPI_Allreduce(MPI_IN_PLACE, &iterPop, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &acceptedMoves, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &totalMoves, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
    double acceptedFrac = acceptedMoves / totalMoves;

    //all procs have the same Eshift
    Eshift -= (0.1 / tau) * std::log(iterPop / olditerPop);
    olditerPop = iterPop;

    if (iter % nGeneration == 0 && iter != 0)
    {
      Stats.Block();
      double rt = Stats.BlockCorrTime();
      double Neff = Stats.Neff();
      //accumulate the total energy across processors
#ifndef SERIAL
      MPI_Allreduce(MPI_IN_PLACE, &rt, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &Neff, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &E2num, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &Enum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &Eden, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      rt /= commsize;
#endif
      //calculate average energy across all procs
      double Ecurrentavg = Enum / Eden;
      Enum /= Eden;
      E2num /= Eden;
      double variance = E2num - Enum * Enum;
      double error = std::sqrt(rt * variance / Neff);

      //this is the exponentially moving average
      if (iter == nGeneration) { Eexp = Ecurrentavg; }
      else { Eexp = 0.9 * Eexp + 0.1 * Ecurrentavg; }

      //this is the average energy, but only start
      //taking averages after 4 generations because 
      //we assume that it takes atleast 4 generations to reach equilibration
      if (iter/nGeneration == 4) { Eavg = Ecurrentavg; }
      else if (iter/nGeneration > 4)
      {
	    int oldIter = iter/nGeneration;
	    Eavg = ((oldIter - 4) * Eavg + Ecurrentavg) / (oldIter - 3);
      }
      else { Eavg = Eexp; }

      if (commrank == 0) {
	    cout << format("%8i %14.8f (%8.2e) %14.8f %14.8f %8.3f %10i %8.2f %14.8f %8.2f\n") % iter % Ecurrentavg % error % Eexp % Eavg % acceptedFrac % Walkers.size() % (iterPop/commsize) % Eshift % (getTime()-startofCalc);
      }

      //restart Enum and Eden
      Enum = 0.0; Eden = 0.0; E2num = 0.0;
    }

    //reconfigure
    reconfigure(Walkers);
  }
}


#endif
