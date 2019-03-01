/** \file atom_typer.h
 *
 *  Classes and routines for reducing an atom down to a numerical type or vector.
 *
 *  Created on: Feb 27, 2019
 *      Author: dkoes
 */

#ifndef ATOMTYPER_H_
#define ATOMTYPER_H_

#include <openbabel/atom.h>
#include <openbabel/elements.h>
#include <vector>
#include <memory>
#include <iostream>
#include <unordered_map>

namespace libmolgrid {

/***** Base classes **********/

/** \brief Base class for generating numerical types along with atomic radius */
class AtomIndexTyper {
  public:
    AtomIndexTyper();
    virtual ~AtomIndexTyper();

    /// return number of types
    virtual unsigned num_types() const;

    ///return type index of a along with the apprioriate index
    virtual std::pair<int,float> get_type(OpenBabel::OBAtom& a) const = 0;

    //return vector of string representations of types
    //this isn't expected to be particularly efficient
    virtual std::vector<std::string> get_type_names() const = 0;
};

/** \brief Base class for generating vector types */
class AtomVectorTyper {
  public:
    AtomVectorTyper() {}
    virtual ~AtomVectorTyper() {}

    /// return number of types
    virtual unsigned num_types() const = 0;

    ///set vector type of atom a, return radius
    virtual float get_type(OpenBabel::OBAtom& a, std::vector<float>& typ) const = 0;

    //return vector of string representations of types
    //this isn't expected to be particularly efficient
    virtual std::vector<std::string> get_type_names() const = 0;

};

/** \brief Base class for mapping between type indices */
class AtomIndexTypeMapper {
  public:
    AtomIndexTypeMapper() {}
    virtual ~AtomIndexTypeMapper() {}

    /// return number of mapped types, zero if unknown (no mapping)
    virtual unsigned num_types() const { return 0; };

    /// return mapped type
    virtual int get_type(unsigned origt) const { return origt; }
};


/*********** Atom typers *****************/

/** \brief Calculate gnina types
 *
 * These are variants of AutoDock4 types. */
class GninaIndexTyper: public AtomIndexTyper {
    bool use_covalent = false;

  public:

    enum type {
      /* 0 */Hydrogen, // H_H_X,
      /* 1 */PolarHydrogen,//(can donate) H_HD_X,
      /* 2 */AliphaticCarbonXSHydrophobe,// C_C_C_H,  //hydrophobic according to xscale
      /* 3 */AliphaticCarbonXSNonHydrophobe,//C_C_C_P, //not hydrophobic (according to xs)
      /* 4 */AromaticCarbonXSHydrophobe,//C_A_C_H,
      /* 5 */AromaticCarbonXSNonHydrophobe,//C_A_C_P,
      /* 6 */Nitrogen,//N_N_N_P, no hydrogen bonding
      /* 7 */NitrogenXSDonor,//N_N_N_D,
      /* 8 */NitrogenXSDonorAcceptor,//N_NA_N_DA, also an autodock acceptor
      /* 9 */NitrogenXSAcceptor,//N_NA_N_A, also considered an acceptor by autodock
      /* 10 */Oxygen,//O_O_O_P,
      /* 11 */OxygenXSDonor,//O_O_O_D,
      /* 12 */OxygenXSDonorAcceptor,//O_OA_O_DA, also an autodock acceptor
      /* 13 */OxygenXSAcceptor,//O_OA_O_A, also an autodock acceptor
      /* 14 */Sulfur,//S_S_S_P,
      /* 15 */SulfurAcceptor,//S_SA_S_P, XS doesn't do sulfur acceptors
      /* 16 */Phosphorus,//P_P_P_P,
      /* 17 */Fluorine,//F_F_F_H,
      /* 18 */Chlorine,//Cl_Cl_Cl_H,
      /* 19 */Bromine,//Br_Br_Br_H,
      /* 20 */Iodine,//I_I_I_H,
      /* 21 */Magnesium,//Met_Mg_Met_D,
      /* 22 */Manganese,//Met_Mn_Met_D,
      /* 23 */Zinc,// Met_Zn_Met_D,
      /* 24 */Calcium,//Met_Ca_Met_D,
      /* 25 */Iron,//Met_Fe_Met_D,
      /* 26 */GenericMetal,//Met_METAL_Met_D,
      /* 27 */Boron,//there are 160 cmpds in pdbbind (general, not refined) with boron
      NumTypes
    };

    //Create a gnina typer.  If usec is true, use the gnina determined covalent radius.
    GninaIndexTyper(bool usec = false): use_covalent(usec) {}
    virtual ~GninaIndexTyper() {}

    /// return number of types
    virtual unsigned num_types() const;

    ///return type index of a
    virtual std::pair<int,float> get_type(OpenBabel::OBAtom& a) const;

    //return vector of string representations of types
    virtual std::vector<std::string> get_type_names() const;
};

/** \brief Calculate element types
 *
 * There are quite a few elements, so should probably run this through
 * an organic chem atom mapper that reduces to number of types.
 * The type id is the atomic number.  Any element with atomic number
 * greater than the specified max is assigned type zero.
 *  */
class ElementIndexTyper: public AtomIndexTyper {
    unsigned last_elem;
  public:
    ElementIndexTyper(unsigned maxe = 84): last_elem(maxe) {}
    virtual ~ElementIndexTyper() {}

    /// return number of types
    virtual unsigned num_types() const;

    ///return type index of a
    virtual std::pair<int,float> get_type(OpenBabel::OBAtom& a) const;

    //return vector of string representations of types
    virtual std::vector<std::string> get_type_names() const;
};

/** \brief Wrap an atom typer with a mapper
 *
 */
template<class Mapper, class Typer>
class MappedAtomIndexTyper: public AtomIndexTyper {
    Mapper mapper;
    Typer typer;
  public:
    MappedAtomIndexTyper(const Mapper& map, const Typer& typr): mapper(map), typer(typr) {}
    virtual ~MappedAtomIndexTyper() {}

    /// return number of types
    virtual unsigned num_types() const;

    ///return type index of a
    virtual std::pair<int,float> get_type(OpenBabel::OBAtom& a) const;

    //return vector of string representations of types
    virtual std::vector<std::string> get_type_names() const;
};


/** \brief Decompose gnina types into elements and properties.  Result is boolean.
 *
 * Hydrophobic, Aromatic, Donor, Acceptor
 *
 * These are variants of AutoDock4 types. */
class GninaVectorTyper: public AtomVectorTyper {
  public:
    GninaVectorTyper() {}
    virtual ~GninaVectorTyper() {}

    /// return number of types
    virtual unsigned num_types() const;

    ///return type index of a
    virtual float get_type(OpenBabel::OBAtom& a, std::vector<float>& typ) const;

    //return vector of string representations of types
    virtual std::vector<std::string> get_type_names() const;
};


/*********** Atom mappers *****************/

/** \brief Map atom types based on provided file.
 *
 * Each line for the provided file specifies a single type.
 * Types are specified using type names.
 * This class must be provided the type names properly indexed (should match get_type_names).
 */
class FileAtomMapper : public AtomIndexTypeMapper {
    void setup(std::istream& in, const std::vector<std::string>& type_names);
  public:

    ///initialize from filename
    FileAtomMapper(std::string& fname, const std::vector<std::string>& type_names);

    ///initialize from stream
    FileAtomMapper(std::istream& in, const std::vector<std::string>& type_names);

    virtual ~FileAtomMapper() {}

    /// return number of mapped types, zero if unknown (no mapping)
    virtual unsigned num_types() const;

    /// return mapped type
    virtual int get_type(unsigned origt) const;
};

/** \brief Map atom types onto a provided subset.
 */
class SubsetAtomMapper: public AtomIndexTypeMapper {
    std::unordered_map<int, int> old2new;
    int default_type; // if not in map
  public:
    /// Indices of map are new types, values are the old types,
    /// if include_catchall is true, the last type will be the type
    /// returned for anything not in map (otherwise -1 is returned)
    SubsetAtomMapper(const std::vector<int>& map, bool include_catchall=true);

    ///surjective mapping
    SubsetAtomMapper(const std::vector< std::vector<int> >& map, bool include_catchall=true);

    /// return number of mapped types, zero if unknown (no mapping)
    virtual unsigned num_types() const;

    /// return mapped type
    virtual int get_type(unsigned origt) const;
};

} /* namespace libmolgrid */

#endif /* ATOMTYPER_H_ */
