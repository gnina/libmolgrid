/*
 * atom_typer.cpp
 *
 *  Routines for typing and mapping atoms.
 *
 *  Created on: Feb 27, 2019
 *      Author: dkoes
 */

#include <atom_typer.h>
#include <openbabel/obiter.h>

using namespace OpenBabel;
using namespace std;

namespace libmolgrid {

/**************  GninaIndexTyper  ********************/

/// return number of types
unsigned GninaIndexTyper::num_types() const {
  return NumTypes;
}

///return type index of a
std::pair<int,float> GninaIndexTyper::get_type(OpenBabel::OBAtom& a) const {

  bool hbonded = false;
  bool heteroBonded = false;


  FOR_NBORS_OF_ATOM(neigh, a){
    if (neigh->GetAtomicNum() == 1)
      hbonded = true;
    else if (neigh->GetAtomicNum() != 6)
      heteroBonded = true; //hetero anything that is not hydrogen and not carbon
  }

  int ret = -1;
  switch (a.GetAtomicNum()) {
  case 1: //H
    ret = a.IsPolarHydrogen() ? PolarHydrogen : Hydrogen;
    break;
  case 5: //B(oron)
    ret = Boron;
    break;
  case 6: //C
    if (a.IsAromatic()) {
      ret =
          heteroBonded ?
                         AromaticCarbonXSNonHydrophobe :
                         AromaticCarbonXSHydrophobe;
    } else { //aliphatic
      ret =
          heteroBonded ?
                         AliphaticCarbonXSNonHydrophobe :
                         AliphaticCarbonXSHydrophobe;
    }
    break;
  case 7: //N
    if (a.IsHbondAcceptor()) {
      ret = hbonded ? NitrogenXSDonorAcceptor : NitrogenXSAcceptor;
    } else {
      ret = hbonded ? NitrogenXSDonor : Nitrogen;
    }
    break;
  case 8: //O
    if (a.IsHbondAcceptor()) {
      ret = hbonded ? OxygenXSDonorAcceptor : OxygenXSAcceptor;
    } else {
      ret = hbonded ? OxygenXSDonor : Oxygen;
    }
    break;
  case 9: //F
    ret = Fluorine;
    break;
  case 12: //Mg
    ret = Magnesium;
    break;
  case 15: //P
    ret = Phosphorus;
    break;
  case 16: //S
    ret = a.IsHbondAcceptor() ? SulfurAcceptor : Sulfur;
    break;
  case 17: //Cl
    ret = Chlorine;
    break;
  case 20: // Ca
    ret = Calcium;
    break;
  case 25: // Mn
    ret = Manganese;
    break;
  case 26: // Fe
    ret = Iron;
    break;
  case 30: // Zn
    ret = Zinc;
    break;
  case 35: //Br
    ret = Bromine;
    break;
  case 53: //I
    ret = Iodine;
    break;
  default:
    ret = GenericMetal;
  }

  return ret;

}

//return vector of string representations of types
std::vector<std::string> GninaIndexTyper::get_type_names() const {

}

/************** Element IndexTyper  ********************/

/// return number of types
unsigned ElementIndexTyper::num_types() const {
}

///return type index of a
std::pair<int,float> ElementIndexTyper::get_type(OpenBabel::OBAtom& a) const {
}

//return vector of string representations of types
std::vector<std::string> ElementIndexTyper::get_type_names() const {
}

/****************  MappedAtomIndexTyper ********************/
template<class Mapper, class Typer>
unsigned MappedAtomIndexTyper::num_types() const {
}

///return type index of a
template<class Mapper, class Typer>
std::pair<int,float> MappedAtomIndexTyper::get_type(OpenBabel::OBAtom& a) const {
}

//return vector of string representations of types
template<class Mapper, class Typer>
std::vector<std::string> MappedAtomIndexTyper::get_type_names() const;
}
;

/****************** GninaVectorTyper ******************/

/// return number of types
unsigned GninaVectorTyper::num_types() const {
}

///return type index of a
float GninaVectorTyper::get_type(OpenBabel::OBAtom& a, std::vector<float>& typ) const {
}

//return vector of string representations of types
std::vector<std::string> GninaVectorTyper::get_type_names() const {
}

/*********** FileAtomMapper *****************/

/// read in map
void FileAtomMapper::setup(std::istream& in) {
}

FileAtomMapper::FileAtomMapper(std::string& fname) {
}

FileAtomMapper::FileAtomMapper(std::istream& in) {
}

/// return number of mapped types, zero if unknown (no mapping)
unsigned FileAtomMapper::num_types() const {
}

/// return mapped type
int FileAtomMapper::get_type(unsigned origt) const {
}

/*************** SubsetAtomMapper **********************/
SubsetAtomMapper::SubsetAtomMapper(const std::vector<int>& map,
    bool include_catchall) {
}

///surjective mapping
SubsetAtomMapper::SubsetAtomMapper(const std::vector<std::vector<int> >& map,
    bool include_catchall) {
}

/// return number of mapped types, zero if unknown (no mapping)
unsigned SubsetAtomMapper::num_types() const {
}

/// return mapped type
int SubsetAtomMapper::get_type(unsigned origt) const {

}

} /* namespace libmolgrid */
