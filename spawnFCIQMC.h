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
#ifndef spawnFCIQMC_HEADER_H
#define spawnFCIQMC_HEADER_H

#include "global.h"
#include "walkersFCIQMC.h"
#include <iostream>
#include <vector>
#include <boost/serialization/serialization.hpp>
#include <Eigen/Dense>

class Determinant;

template <typename T, typename Compare>
std::vector<std::size_t> sort_permutation(int const nDets, const std::vector<T>& vec, Compare compare)
{
  std::vector<std::size_t> p(nDets);
  std::iota(p.begin(), p.end(), 0);
  std::sort(p.begin(), p.end(),
    [&](std::size_t i, std::size_t j){ return compare(vec[i], vec[j]); });
  return p;
}

template <typename T>
void apply_permutation(int const nDets, const std::vector<T>& vec, std::vector<T>& sorted_vec, const std::vector<std::size_t>& p)
{
  std::transform(p.begin(), p.end(), sorted_vec.begin(),
    [&](std::size_t i){ return vec[i]; });
}

using namespace std;

// Class for spawning arrays needed in FCIQMC
class spawnFCIQMC {

 public:
  // The number of determinants spawned to
  int nDets;
  // The list of determinants spawned to
  vector<Determinant> dets;
  // Temporary space for communication and sorting
  vector<Determinant> detsTemp;
  // The amplitudes of spawned walkers
  vector<double> amps;
  vector<double> ampsTemp;

  spawnFCIQMC(int spawnSize) {
    nDets = 0;
    dets.resize(spawnSize);
    amps.resize(spawnSize, 0.0);
    detsTemp.resize(spawnSize);
    ampsTemp.resize(spawnSize, 0.0);
  }

  // Send spawned walkers to their correct processor
  void communicate() {
    // For now, just copy walkers across to the temporary array
    // TODO: Implement parallel FCIQMC
    for (int i=0; i<nDets; i++) {
      detsTemp[i] = dets[i];
      ampsTemp[i] = amps[i];
    }
  }
  
  // Merge multiple spawned walkers to the same determinant, so that each
  // determinant only appears once
  void compress() {

    if (nDets > 0) {
      // Perform sort
      auto p = sort_permutation(nDets, detsTemp, [](Determinant const& a, Determinant const& b){ return (a < b); });
      apply_permutation( nDets, detsTemp, dets, p );
      apply_permutation( nDets, ampsTemp, amps, p );
  
      bool exitOuter = false;
      int j = 0, k = 0;

      // Now the array is sorted, loop through and merge repeats
      while (true) {
        dets[j] = dets[k];
        amps[j] = amps[k];
        while (true) {
          k += 1;
          if (k == nDets) {
            exitOuter = true;
            break;
          }
          if ( dets[j] == dets[k] ) {
            amps[j] += amps[k];
          } else {
            break;
          }
        }
  
        if (exitOuter) break;
        
        if (j == nDets-1) {
          break;
        } else {
          j += 1;
        }
      }
      nDets = j+1;
    }
  }
  
  // Move spawned walkers to the provided main walker list
  void mergeIntoMain(walkersFCIQMC& walkers, double& minPop) {
    int pos;
  
    for (int i = 0; i<nDets; i++) {
      // Is this spawned determinant already in the main list?
      if (walkers.ht.find(dets[i]) != walkers.ht.end()) {
        int iDet = walkers.ht[dets[i]];
        double oldAmp = walkers.amps[iDet];
        double newAmp = amps[i] + oldAmp;
        walkers.amps[iDet] = newAmp;
      }
      else
      {
        // New determinant:

        // Check if a determinant has become unoccupied in the existing list
        // If not, then increase the walkers.ndets by 1 and add it on the end
        if (walkers.firstEmpty <= walkers.lastEmpty) {
          pos = walkers.emptyDets[walkers.firstEmpty];
          walkers.firstEmpty += 1;
        }
        else
        {
          pos = walkers.nDets;
          walkers.nDets += 1;
        }
        walkers.dets[pos] = dets[i];
        walkers.amps[pos] = amps[i];
        walkers.ht[dets[i]] = pos;
      }
    }
  }

};
#endif
