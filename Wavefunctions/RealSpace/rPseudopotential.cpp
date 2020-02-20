#include "rPseudopotential.h"
#include <map>
#include <vector>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <boost/serialization/map.hpp>

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
