#ifndef rPP_HEADER_H
#define rPP_HEADER_H

#include "rDeterminants.h"
#include <Eigen/Dense>
#include <map>
#include <vector>
#include <fstream>
#include <ostream>
#include <boost/algorithm/string.hpp>
#include <boost/serialization/map.hpp>


//this is an object that contains all the information of a pseudopotential for a specific atom
class ppHelper
{
    private:
    friend class Pseudopotential;
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
        ar & _ncore;
        ar & lchannels;
        ar & Indices;
    }

    int _ncore; //number of core electrons replaced in atom
    std::map<int, std::vector<double>> lchannels; //map for different angular momentum channels, -1 corresponds to local part of pseudopotential, 0 is S, 1 is P, 2 is D, etc.
    std::vector<int> Indices; //indices of atom in Ncharge/Ncoord ie. if Carbon dimer with pseudopotential, this would have {0, 1}

    public:
    int ncore() const { return _ncore; }
    inline const std::vector<double> &operator[](int l) const { return lchannels.at(l); }
    inline const std::vector<int> &indices() const { return Indices; }

    auto begin() { return lchannels.begin(); }
    auto end() { return lchannels.end(); }
    auto begin() const { return lchannels.begin(); }
    auto end() const { return lchannels.end(); }
    std::size_t size() { return lchannels.size(); }
};

//this is an object that stores a map for the pseudopotential, the key is the atomic number of the atom and the value is the above object
class Pseudopotential
{
    private:
    friend std::ostream &operator<<(std::ostream &os, const Pseudopotential &PP);
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) { ar & Store; }

    std::map<int, ppHelper> Store; //map for different atoms, key in map is atomic number of atom

    public:
    Pseudopotential(std::string filename = "ppInfo.txt");

    int ncore() const; //returns the total number of core electrons replaced by PP, (ie. this is summed over all atoms)
    const ppHelper &operator[](int atm) const { return Store.at(atm); }

    std::size_t size() const { return Store.size(); }
    auto find(int atm) { return Store.find(atm); }
    auto begin() { return Store.begin(); }
    auto end() { return Store.end(); }
    auto begin() const { return Store.begin(); }
    auto end() const { return Store.end(); }

    //computes the local part of the pseudopotential
    double localPotential(const rDeterminant &d) const;

    //computes matrix elements and coordinates for nonlocal part of pseudopotential for a given electron
    //vq = <q|Vnl|r>, matrix element of nonlocal potential at every quarature point [q]
    //rq = q, coordinate q corresponding to every viq matrix element [q]
    void nonLocalPotential(int i, const rDeterminant &d, std::vector<double> &vq, std::vector<Eigen::Vector3d> &rq) const;

    //computes matrix elements and coordinates for nonlocal part of pseudopotential
    //viq = <q|Vnl|r>_i, matrix element of nonlocal potential at every quarature point [i=elec][q]
    //riq = q_i, coordinate q corresponding to every viq matrix element [i=elec][q]
    void nonLocalPotential(const rDeterminant &d, std::vector<std::vector<double>> &viq, std::vector<std::vector<Eigen::Vector3d>> &riq) const;
};
#endif
