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
double applyPropogatorMetropolis(Wfn &w, Walker &walk, double &wt, double tau, double Eshift, double &Eloc, double E, double Var)
{
  auto random = std::bind(std::uniform_real_distribution<double>(0, 1), std::ref(generator));
  int nelec = walk.d.nelec;

  //calculate local energy before move
  double Elocp = w.rHam(walk);
  ////branching
  double deltap = E - Elocp;
  double signp = std::copysign(1.0, deltap);

  double Ecutp = std::min(std::abs(deltap), 10.0 * std::sqrt(Var)) * signp;
  //scale factor
  double V2p = 0.0;
  for (int i = 0; i < nelec; i++)
  {
    Vector3d v;
    walk.getGradient(i, v, w.getRef());
    V2p += v.squaredNorm(); 
  }
  double factorp = tau * V2p / double(nelec);
  double Sp = Eshift - E + Ecutp / (1.0 + factorp * factorp);

  //double Ecutp = std::min(std::abs(deltap), 0.2 * std::sqrt(nelec / tau)) * signp;
  //double Sp = Eshift - E + Ecutp;

  //size consistent (really size extensive) t-moves
  if (schd.doTMove)
  {
    const Pseudopotential &pp = *schd.pseudo;
    for (int i = 0; i < nelec; i++)
    {
      std::vector<double> vq;
      std::vector<Eigen::Vector3d> rq;
      pp.nonLocalPotential(i, walk.d, vq, rq); 

      //make forward matrix elements
      std::vector<double> tf;
      std::vector<Eigen::Vector3d> rf;
      for (int q = 0; q < vq.size(); q++)
      {
        double yq = w.getOverlapFactor(i, rq.at(q), walk);
        double tq = - tau * vq.at(q) * yq;
        if (tq > 0)
        {
          tf.push_back(tq);
          rf.push_back(rq.at(q));
        }
      }

      //heat bath move
      double cumT = 0.0;
      std::vector<double> T;
      //first possible move is no move, with txx = 1.0
      cumT += 1.0;
      T.push_back(cumT);
      for (int q = 0; q < tf.size(); q++)
      {
        cumT += tf.at(q);
        T.push_back(cumT);
      }
      //get index of move, remember first index is no move
      double move = random() * cumT;
      int index = std::lower_bound(T.begin(), T.end(), move) - T.begin() - 1;

      //accept move?
      if (index < 0) { continue; } //no move
      else //calculate acceptance probability
      {
        double y0 = w.getOverlapFactor(i, rf.at(index), walk);

        //forward acceptance
        double af = 1.0 + std::accumulate(tf.begin(), tf.end(), 0.0);

        //make reverse matrix elements
        rDeterminant d = walk.d;
        d.coord[i] = rf.at(index);
        pp.nonLocalPotential(i, d, vq, rq); 

        std::vector<double> tr;
        for (int q = 0; q < vq.size(); q++)
        {
          double yq = w.getOverlapFactor(i, rq.at(q), walk) / y0;
          double tq = - tau * vq.at(q) * yq;
          if (tq > 0)
          {
            tr.push_back(tq);
          }
        }

        //reverse acceptance
        double ar = 1.0 + std::accumulate(tr.begin(), tr.end(), 0.0);

        double p = std::min(af / ar, 1.0);
        if (p > random()) { walk.updateWalker(i, rf.at(index), w.getRef(), w.getCorr()); }
      }//accept
    }//nelec
  }

  //diffusion
  //propogate every electron once via metropolis-hastings with dmc propogator
  double acceptedProb = 0.0;
  double dr2_accepted = 0.0, dr2_proposed = 0.0;
  double pavg = 0.0;
  for (int i = 0; i < nelec; i++)
  {
    double ovlpRatio = 0.0, proposalProb = 0.0;
    Vector3d step;
    walk.doDMCMove(step, i, tau, w.getRef(), w.getCorr(), ovlpRatio, proposalProb);
    Vector3d newCoord = step + walk.d.coord[i];
    dr2_proposed += step.squaredNorm();

    //accept move?
    double ovlpProb = ovlpRatio * ovlpRatio;
    double p = std::min(ovlpProb * proposalProb, 1.0);
    pavg += p;
    if (p > random() && ovlpRatio > 0.0) //accept move based on metropolis criterion and if node is not crossed
    {
      acceptedProb++;
      walk.updateWalker(i, newCoord, w.getRef(), w.getCorr());
      dr2_accepted += step.squaredNorm();
    }
  }
  acceptedProb /= double(nelec);
  double tau_eff = tau * dr2_accepted / dr2_proposed; //effective time step due to metropolis algorithm
  pavg /= double(nelec);

  //calculate local energy after move
  Eloc = w.rHam(walk);
  double delta = E - Eloc;
  double sign = std::copysign(1.0, delta);

  double Ecut = std::min(std::abs(delta), 10.0 * std::sqrt(Var)) * sign;
  //scale factor
  double V2 = 0.0;
  for (int i = 0; i < nelec; i++)
  {
    Vector3d v;
    walk.getGradient(i, v, w.getRef());
    V2 += v.squaredNorm(); 
  }
  double factor = tau * V2 / double(nelec);
  double S = Eshift - E + Ecut / (1.0 + factor * factor);

  //double Ecut = std::min(std::abs(delta), 0.2 * std::sqrt(nelec / tau)) * sign;
  //double S = Eshift - E + Ecut;

  //branching
  double arg = (1.0 - pavg / 2.0) * Sp + (pavg / 2.0) * S;
  //double arg = 0.5 * (Sp + S);
  wt *= std::exp(tau_eff * arg);

  if (wt > 10)
  {
    cout << "Process: " << commrank << endl;
    cout << "weight: " << wt << endl;
    cout << "delta weight: " << std::exp(tau_eff * arg) << endl;
    cout << "tau effective: " << tau_eff << endl;
    cout << "delta weight arg: " << arg << endl;
    cout << "Eloc before diffusion: " << Elocp << endl;
    cout << "Eloc after diffusion: " << Eloc << endl;
    cout << "coords" << endl;
    cout << walk.d << endl;
    cout << "Rij" << endl;
    cout << walk.Rij << endl;
    cout << "RiN" << endl;
    cout << walk.RiN << endl;
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

  //text file of run { iter, time, Energy of population }
  ofstream f("simulation.out");
  f << std::fixed;
  f << std::setprecision(5);

  //total weight equals to schd.nwalk;
  double oldwt = schd.nwalk;
  double targetPop = schd.nwalk;

  //energy of inital population
  double nE0 = 0.0;
  double nE20 = 0.0;
  double dE0 = 0.0;  
  for (auto it = Walkers.begin(); it != Walkers.end(); it++)
  {
      double Eloc = w.rHam(it->first);
      double wt = it->second;
      nE0 += Eloc * wt;
      nE20 += Eloc * Eloc * wt;
      dE0 += wt;
  }
  double E0 = nE0 / dE0;
  double E20 = nE20 / dE0;
#ifndef SERIAL
      MPI_Allreduce(MPI_IN_PLACE, &E0, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &E20, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
  E0 /= commsize;
  E20 /= commsize;
  double Var0 = E20 - E0 * E0;
  if (commrank == 0) { f << 0 << " " << double(0) << " " << E0 << " " << Var0 << " " << Walkers.size() << endl; }
  if (commrank == 0) { cout << "Energy of initial population: " << E0 << endl; }

  //run calculation for niter iterations
  int niter = schd.maxIter;

  //time step
  double tau = schd.tau;

  //number of iterations in a generation
  int nGeneration = schd.nGeneration;

  //energy average
  double E = E0;
  double Var = Var0;
  double Eavg = 0.0, Eexp = 0.0;
  Statistics StatsE;
  Statistics StatsE2;

  double olditerPop = oldwt;
  for (int iter = 0; iter < niter + 1; iter++)
  {
    //propogate walkers time tau and calculate energy
    double N = 0.0, N2 = 0.0;
    double D = 0.0;
    double iterPop = 0.0;
    double acceptedMoves = 0.0;
    for (auto it = Walkers.begin(); it != Walkers.end(); it++)
    {
      Walker &walk = it->first;
      double &wt = it->second;

      double Eloc = 0.0;
      acceptedMoves += applyPropogatorMetropolis(w, walk, wt, tau, Eshift, Eloc, E, Var);

      iterPop += wt;
      N += wt * Eloc;
      N2 += wt * Eloc * Eloc;
      D += wt;
    }

    //energy of population at time t
    double Et = N / D;
    double Et2 = N2 / D;
    double Vart = Et2 - Et * Et;
    StatsE.push_back(Et);
    StatsE2.push_back(Et2);

    //average some variables over processes
    double totalMoves = double(Walkers.size());
    double totalPop = iterPop;
    //VectorXd npp = VectorXd::Zero(commsize);
    //npp(commrank) = totalMoves;
    //VectorXd wpp = VectorXd::Zero(commsize);
    //wpp(commrank) = iterPop;
    //VectorXd Eshiftpp = VectorXd::Zero(commsize);
    //Eshiftpp(commrank) = Eshift;
#ifndef SERIAL
    MPI_Allreduce(MPI_IN_PLACE, &Et, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &Vart, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &acceptedMoves, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &totalMoves, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &totalPop, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    //MPI_Allreduce(MPI_IN_PLACE, &npp(0), npp.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    //MPI_Allreduce(MPI_IN_PLACE, &wpp(0), wpp.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    //MPI_Allreduce(MPI_IN_PLACE, &Eshiftpp(0), Eshiftpp.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
    Et /= commsize;
    Vart /= commsize;
    double acceptedFrac = acceptedMoves / totalMoves;

    E = 0.9 * E + 0.1 * Et;
    Var = 0.9 * Var + 0.1 * Vart;
    if (commrank == 0) { f << iter + 1 << " " << tau * double(iter + 1) << " " << Et << " " << Vart << " " << int(totalMoves / commsize) << " " << totalPop / commsize << endl; }
    if (commrank == 0) { f << "\t" << E << " " << Var << endl; }
    //if (commrank == 0) { f << npp.transpose() << endl; }
    //if (commrank == 0) { f << wpp.transpose() << endl; }
    //if (commrank == 0) { f << Eshiftpp.transpose() << endl << endl; }

    //every process is independent and have different Eshifts
    Eshift = E - (0.1 / tau) * std::log(iterPop / targetPop);

    //incase of catastrophic failure
    if (iterPop > 2 * targetPop)
    {
      cout << "population explosion" << endl;
      cout << commrank << endl;
      cout << Eshift << endl;
      cout << iterPop << " / " << targetPop << endl;
      cout << Walkers.size() << " / " << targetPop << endl;
      exit(0);
    }

    if (iter % nGeneration == 0 && iter != 0)
    {
      //calculate average energy across generation
      double Ep = StatsE.Average();
      StatsE.Block();
      double tp = StatsE.BlockCorrTime();
      double Ep2 = StatsE2.Average();
      double Varp = tp * (Ep2 - Ep * Ep);
      double Neff = StatsE.Neff();
      StatsE = Statistics(); //restart statistics for new generation
      StatsE2 = Statistics(); //restart statistics for new generation
      double Eshiftavg = Eshift;
      //now average over processes
#ifndef SERIAL
    MPI_Allreduce(MPI_IN_PLACE, &Ep, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &Varp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &tp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &Neff, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &Eshiftavg, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
      double Ebar = Ep / commsize;
      double Varbar = Varp / commsize / commsize;
      double tbar = tp / commsize;
      double error = std::sqrt(Varbar / Neff);
      Eshiftavg /= commsize;

      E = Ebar;
      Var = Varbar;

      //this is the exponentially moving average over generations
      if (iter == nGeneration) { Eexp = Ebar; }
      else { Eexp = 0.9 * Eexp + 0.1 * Ebar; }

      //this is the average energy over generations, but only start
      //taking averages after 4 generations because 
      //we assume that it takes atleast 4 generations to reach equilibration
      if (iter/nGeneration == 4) { Eavg = Ebar; }
      else if (iter/nGeneration > 4)
      {
	    int oldIter = iter/nGeneration;
	    Eavg = ((oldIter - 4) * Eavg + Ebar) / (oldIter - 3);
      }
      else { Eavg = Eexp; }

      if (commrank == 0) {
	    cout << format("%8i %14.8f (%8.2e) %8.1f %14.8f %14.8f %8.3f %10i %8.2f %14.8f %8.2f\n") % iter % Ebar % error % tbar % Eexp % Eavg % acceptedFrac % int(totalMoves/commsize) % (totalPop/commsize) % Eshiftavg % (getTime()-startofCalc);
      }

    }

    //reconfigure
    reconfigure(Walkers);
  }
}


#endif
