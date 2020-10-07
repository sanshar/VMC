#include "rPseudopotential.h"
#include "input.h"
#include <map>
#include <vector>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <boost/serialization/map.hpp>
#include <boost/math/special_functions/legendre.hpp>

std::ostream &operator<<(std::ostream &os, const Pseudopotential &PP)
{
    for (auto it = PP.begin(); it != PP.end(); ++it)
    {
        int atm = it->first;
        const ppHelper &ppatm = it->second;
        os << atm << "\t" << ppatm.ncore() << "\tidx";
        for (int i = 0; i < ppatm.indices().size(); i++) { os << "\t" << ppatm.indices()[i]; }
        os << std::endl;
        for (auto it1 = ppatm.begin(); it1 != ppatm.end(); ++it1)
        {
            int l = it1->first;
            const std::vector<double> &lchannel = it1->second;
            os << l << std::endl;
            for (int i = 0; i < lchannel.size(); ++i)
            {
                if (i % 3 == 0 && i != 0) { os << std::endl; }
                os << lchannel[i] << "\t";
            }    
            os << std::endl;
        }
    }
    return os;
}

int Pseudopotential::ncore() const
{
    int ncore = 0;
    if (!Store.empty())
    {
        for (auto it = begin(); it != end(); ++it)
        {
            int atm = it->first;
            const ppHelper &ppatm = it->second;
            ncore += ppatm.ncore();
        }
    }
    return ncore;
}

Pseudopotential::Pseudopotential(std::string filename)
{
    std::ifstream f(filename);
    if (!f.is_open()) { Store.clear(); } //if no file, empty PP
    else
    {
        int atm, l;
        while(f.good())
        {
            std::string line;
            std::getline(f, line);
            boost::algorithm::trim(line);
            std::vector<std::string> token;
            boost::split(token, line, boost::is_any_of(" \t\n"), boost::token_compress_on);
            
            if (token.size() == 1 && token[0] == "\0") { break; } //reached end of file

            if (token.size() > 3) //new atom
            {
                atm = std::stoi(token[0]);
                ppHelper ppatm;
                ppatm._ncore = std::stoi(token[1]);
                Store[atm] = ppatm;
                for (int i = 3; i < token.size() && token[i] != "\0"; i++) { Store[atm].Indices.push_back(std::stoi(token[i])); }
            }
            else if (token.size() == 1) // angular momentum channel
            {
                l = std::stoi(token[0]);
                std::vector<double> ppl;
                Store[atm].lchannels[l] = ppl;
            }
            else if (token.size() == 3) // power - exponent - coefficent
            {
                Store[atm].lchannels[l].push_back(std::stod(token[0]));
                Store[atm].lchannels[l].push_back(std::stod(token[1]));
                Store[atm].lchannels[l].push_back(std::stod(token[2]));
            }
        }
    }
}

//computes the local part of the pseudopotential
double Pseudopotential::localPotential(const rDeterminant &d) const
{
    double Vl = 0.0;
    if (size()) //if pseudopotential object is not empty
    {
        for (int i = 0; i < d.nelec; i++) //loop over electrons
        {
            for (auto it = begin(); it != end(); ++it) //loop over atoms with pseudopotential
            {
                const ppHelper &ppatm = it->second;
                for (int a = 0; a < ppatm.indices().size(); a++) //loop over indices of atom
                {
                    int I = ppatm.indices()[a]; //atom index
                    auto it1 = ppatm.begin();
                    int l = it1->first;
                    if (l == -1) //first angular momentum channel should be local potential
                    {
                        //power - exponent - coeff vector
                        const std::vector<double> &pec = it1->second; 

                        //vector
                        Eigen::Vector3d riI = d.coord[i] - schd.Ncoords[I];
  
                        //calculate potential
                        double v = 0.0;
                        for (int m = 0; m < pec.size(); m = m + 3) { v += std::pow(riI.norm(), pec[m] - 2) * std::exp(-pec[m + 1] * riI.squaredNorm()) * pec[m + 2]; }
                        
                        //accumulate
                        Vl += v;    
                    }//if local potential
                }//atm idx
            }//atm
        }//elec
    }//if
    return Vl;
}

//computes matrix elements and coordinates for nonlocal part of pseudopotential
//viq = <q|Vnl|r>_i, matrix element of nonlocal potential at every quarature point [i=elec][q]
//riq = q_i, coordinate q corresponding to every viq matrix element [i=elec][q]
void Pseudopotential::nonLocalPotential(const rDeterminant &d, std::vector<std::vector<double>> &viq, std::vector<std::vector<Eigen::Vector3d>> &riq) const
{
    auto random = std::bind(std::uniform_real_distribution<double>(0, 1), std::ref(generator));
    viq.clear();
    riq.clear();
    if (size()) //if pseudopotential object is not empty
    {
        for (int i = 0; i < d.nelec; i++) //loop over electrons
        {
            std::vector<double> vq;
            std::vector<Vector3d> rq;
            for (auto it = begin(); it != end(); ++it) //loop over atoms with pseudopotential
            {
                const ppHelper &ppatm = it->second;
                for (int a = 0; a < ppatm.indices().size(); a++) //loop over indices of atom
                {
                    int I = ppatm.indices()[a];
                    for (auto it1 = ppatm.begin(); it1 != ppatm.end(); it1++) //loop over angular momentum channels
                    {
                        int l = it1->first; //angular momentum
                        const std::vector<double> &pec = it1->second; //power - exponent - coeff vector
                        Eigen::Vector3d rI = schd.Ncoords[I];
                        Eigen::Vector3d ri = d.coord[i];
                        Eigen::Vector3d riI = ri - rI;
                        
                        //if atom - elec distance larger than 2.0 au, don't calculate nonlocal potential
                        if (l == -1 || riI.norm() > schd.pCutOff) { continue; } 
                        
                        //calculate potential
                        double v = 0.0;
                        for (int m = 0; m < pec.size(); m = m + 3) { v += std::pow(riI.norm(), pec[m] - 2) * std::exp(-pec[m + 1] * riI.squaredNorm()) * pec[m + 2]; }
                        
                        //random rotation
                        Eigen::Vector3d riIhat = riI.normalized();
                        double angle = random() * 2.0 * M_PI;
                        //double angle = 0.0; //this is easier when debugging
                        Eigen::AngleAxis<double> rot(angle, riIhat);
                        
                        double C = (2.0 * double(l) + 1.0);
                        for (int q = 0; q < schd.Q.size(); q++)
                        {
                            //calculate new vector, riprime
                            Eigen::Vector3d riIprime = riI.norm() * (rot * schd.Q[q]);
                            Eigen::Vector3d riprime = riIprime + rI;
                            
                            //calculate angle
                            double costheta = riI.dot(riIprime) / (riI.norm() * riIprime.norm());
                            
                            //legendre polynomial
                            double Pl = boost::math::legendre_p<double>(l, costheta);
                            
                            //update tensors
                            vq.push_back(v * C * Pl / double(schd.Q.size()));
                            rq.push_back(riprime);
                        }//q
                    }//l
                }//atm idx
            }//atm

            //update tensors
            viq.push_back(vq);
            riq.push_back(rq);
        }//elec
    }//if
}
