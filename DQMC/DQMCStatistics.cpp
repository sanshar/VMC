#ifndef SERIAL
#include <iostream>
#include "mpi.h"
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif
#include <boost/format.hpp>
#include "global.h"
#include "DQMCStatistics.h"

using namespace std;
using namespace Eigen;
using namespace boost;

// constructor
DQMCStatistics::DQMCStatistics(int pSampleSize) 
{
  nSamples = 0;
  sampleSize = pSampleSize;
  numMean = ArrayXcd::Zero(sampleSize);
  denomMean = ArrayXcd::Zero(sampleSize);
  denomAbsMean = ArrayXd::Zero(sampleSize);
  num2Mean = ArrayXcd::Zero(sampleSize);
  denom2Mean = ArrayXcd::Zero(sampleSize);
  num_denomMean = ArrayXcd::Zero(sampleSize);
}


// store samples and update running averages
void DQMCStatistics::addSamples(ArrayXcd& numSample, ArrayXcd& denomSample)
{
  numMean += (numSample - numMean) / (nSamples + 1.);
  denomMean += (denomSample - denomMean) / (nSamples + 1.);
  denomAbsMean += (denomSample.abs() - denomAbsMean) / (nSamples + 1.);
  numSamples.push_back(numSample);
  denomSamples.push_back(denomSample);
  nSamples++;
}


// calculates error by blocking data
// use after gathering data across processes for better estimates
void DQMCStatistics::calcError(ArrayXd& error, ArrayXd& error2)
{
  ArrayXcd eneEstimates = numMean / denomMean;
  int nBlocks;
  if (nSamples <= 100) nBlocks = 1;
  else nBlocks = 10;
  size_t blockSize = size_t(nSamples / 10);
  ArrayXd var(sampleSize), var2(sampleSize);
  var.setZero(); var2.setZero();

  // calculate variance of blocked energies on each process
  for (int i = 0; i < nBlocks; i++) {
    ArrayXcd blockNum(sampleSize), blockDenom(sampleSize);
    blockNum.setZero(); blockDenom.setZero();
    for (int n = i * blockSize; n < (i + 1) * blockSize; n++) {
      blockNum += numSamples[n];
      blockDenom += denomSamples[n];
    }
    blockNum /= blockSize;
    blockDenom /= blockSize;
    ArrayXcd blockEne(sampleSize);
    blockEne = blockNum / blockDenom;
    var += (blockEne - eneEstimates).abs().pow(2);
    var2 += (blockEne - eneEstimates).abs().pow(4);
  }
  
  // gather variance across processes
  MPI_Allreduce(MPI_IN_PLACE, var.data(), var.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, var2.data(), var2.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  int nBlockedSamples = nBlocks * commsize;
  var /= (nBlockedSamples - 1);
  var2 /= (nBlockedSamples);

  // calculate error estimates a la clt
  error = sqrt(var / nBlockedSamples);
  error2 = sqrt((var2 - (nBlockedSamples - 3) * var.pow(2) / (nBlockedSamples - 1)) / nBlockedSamples) / 2. / sqrt(var) / sqrt(sqrt(nBlockedSamples));
}
 

// gather data from all the processes and print quantities
// to be used at the end of a calculation
// iTime used only for printing
void DQMCStatistics::gatherAndPrintStatistics(ArrayXd iTime)
{
  // gather data across processes
  MPI_Allreduce(MPI_IN_PLACE, numMean.data(), numMean.size(), MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, denomMean.data(), denomMean.size(), MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, denomAbsMean.data(), denomAbsMean.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  
  numMean /= commsize;
  denomMean /= commsize;
  denomAbsMean /= commsize;
  ArrayXcd eneEstimates = numMean / denomMean;
  ArrayXcd avgPhase = denomMean / denomAbsMean;

  // calc error estimates
  ArrayXd error, error2;
  calcError(error, error2);

  // print
  if (commrank == 0) {
    cout << "          iTime                 Energy                     Energy error         Average phase\n";
    for (int n = 0; n < sampleSize; n++) {
      cout << format(" %14.2f   (%14.8f, %14.8f)   (%8.2e   (%8.2e))   (%3.3f, %3.3f) \n") % iTime(n) % eneEstimates(n).real() % eneEstimates(n).imag() % error(n) % error2(n) % avgPhase(n).real() % avgPhase(n).imag(); 
    }
  }
}


// prints running averages from proc 0
void DQMCStatistics::printStatistics() 
{
  return;
};


// write samples to disk
void DQMCStatistics::writeSamples()
{
  return;
};
