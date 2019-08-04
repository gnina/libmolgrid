/*
 * atom_typer.cpp
 *
 *  Routines for typing and mapping atoms.
 *
 *  Created on: Feb 27, 2019
 *      Author: dkoes
 */

#include "libmolgrid/atom_typer.h"
#include <openbabel/obiter.h>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

#if (OB_VERSION >= OB_VERSION_CHECK(2,4,90))
# include <openbabel/elements.h>
# define GET_SYMBOL OpenBabel::OBElements::GetSymbol
# define GET_COVALENT_RAD OBElements::GetCovalentRad
# define GET_NAME OBElements::GetName
#else
# define GET_SYMBOL etab.GetSymbol
# define GET_COVALENT_RAD etab.GetCovalentRad
# define GET_NAME etab.GetName
#endif

using namespace OpenBabel;
using namespace std;

namespace libmolgrid {

/**************  GninaIndexTyper  ********************/

const GninaIndexTyper::info GninaIndexTyper::default_data[GninaIndexTyper::NumTypes] = { //el, ad, xs
    { Hydrogen,"Hydrogen", "H", 1, 1.000000, 0.020000, 0.000510, 0.000000, 0.370000, 0.37, false, false, false, false},
    { PolarHydrogen, "PolarHydrogen", "HD", 1, 1.000000, 0.020000, 0.000510, 0.000000, 0.370000, 0.370000, false, false, false, false},
    //note we typically use the xs_radius, which assumes a heavy atom-only model
    { AliphaticCarbonXSHydrophobe, "AliphaticCarbonXSHydrophobe", "C", 6, 2.000000, 0.150000, -0.001430, 33.510300, 0.770000, 1.900000, true, false, false, false},
    { AliphaticCarbonXSNonHydrophobe, "AliphaticCarbonXSNonHydrophobe", "C", 6, 2.000000, 0.150000, -0.001430, 33.510300, 0.770000, 1.900000, false, false, false, false},
    { AromaticCarbonXSHydrophobe, "AromaticCarbonXSHydrophobe", "A", 6, 2.000000, 0.150000, -0.000520, 33.510300, 0.770000, 1.900000, true, false, false, false},
    { AromaticCarbonXSNonHydrophobe, "AromaticCarbonXSNonHydrophobe", "A", 6, 2.000000, 0.150000, -0.000520, 33.510300, 0.770000, 1.900000, false, false, false, false},
    { Nitrogen, "Nitrogen", "N", 7, 1.750000, 0.160000, -0.001620, 22.449300, 0.750000, 1.800000, false, false, false, true},
    { NitrogenXSDonor, "NitrogenXSDonor", "N", 7, 1.750000, 0.160000, -0.001620, 22.449300, 0.750000, 1.800000, false, true, false, true},
    { NitrogenXSDonorAcceptor, "NitrogenXSDonorAcceptor", "NA", 7, 1.750000, 0.160000, -0.001620, 22.449300, 0.750000, 1.800000, false, true, true, true},
    { NitrogenXSAcceptor, "NitrogenXSAcceptor", "NA", 7, 1.750000, 0.160000, -0.001620, 22.449300, 0.750000, 1.800000, false, false, true, true},
    { Oxygen, "Oxygen", "O", 8, 1.600000, 0.200000, -0.002510, 17.157300, 0.730000, 1.700000, false, false, false, true},
    { OxygenXSDonor, "OxygenXSDonor", "O", 8, 1.600000, 0.200000, -0.002510, 17.157300, 0.730000, 1.700000, false, true, false, true},
    { OxygenXSDonorAcceptor, "OxygenXSDonorAcceptor", "OA", 8, 1.600000, 0.200000, -0.002510, 17.157300, 0.730000, 1.700000, false, true, true, true},
    { OxygenXSAcceptor, "OxygenXSAcceptor", "OA", 8, 1.600000, 0.200000, -0.002510, 17.157300, 0.730000, 1.700000, false, false, true, true},
    { Sulfur, "Sulfur", "S", 16, 2.000000, 0.200000, -0.002140, 33.510300, 1.020000, 2.000000, false, false, false, true},
    { SulfurAcceptor, "SulfurAcceptor", "SA", 16, 2.000000, 0.200000, -0.002140, 33.510300, 1.020000, 2.000000, false, false, false, true},
    { Phosphorus, "Phosphorus", "P", 15, 2.100000, 0.200000, -0.001100, 38.792400, 1.060000, 2.100000, false, false, false, true},
    { Fluorine, "Fluorine", "F", 9, 1.545000, 0.080000, -0.001100, 15.448000, 0.710000, 1.500000, true, false, false, true},
    { Chlorine, "Chlorine", "Cl", 17, 2.045000, 0.276000, -0.001100, 35.823500, 0.990000, 1.800000, true, false, false, true},
    { Bromine, "Bromine", "Br", 35, 2.165000, 0.389000, -0.001100, 42.566100, 1.140000, 2.000000, true, false, false, true},
    { Iodine, "Iodine", "I", 53, 2.360000, 0.550000, -0.001100, 55.058500, 1.330000, 2.200000, true, false, false, true},
    { Magnesium, "Magnesium", "Mg", 12, 0.650000, 0.875000, -0.001100, 1.560000, 1.300000, 1.200000, false, true, false, true},
    { Manganese, "Manganese", "Mn", 25, 0.650000, 0.875000, -0.001100, 2.140000, 1.390000, 1.200000, false, true, false, true},
    { Zinc, "Zinc", "Zn", 30, 0.740000, 0.550000, -0.001100, 1.700000, 1.310000, 1.200000, false, true, false, true},
    { Calcium, "Calcium", "Ca", 20, 0.990000, 0.550000, -0.001100, 2.770000, 1.740000, 1.200000, false, true, false, true},
    { Iron, "Iron", "Fe", 26, 0.650000, 0.010000, -0.001100, 1.840000, 1.250000, 1.200000, false, true, false, true},
    { GenericMetal, "GenericMetal", "M", 0, 1.200000, 0.000000, -0.001100, 22.449300, 1.750000, 1.200000, false, true, false, true},
    //note AD4 doesn't have boron, so copying from carbon
    { Boron, "Boron", "B", 5, 2.04, 0.180000, -0.0011, 12.052, 0.90, 1.920000, true, false, false, false}
};

/// return number of types
unsigned GninaIndexTyper::num_types() const {
  return NumTypes;
}

std::string GninaIndexTyper::gnina_type_name(int t) {
  if(t >= 0 && t < GninaIndexTyper::NumTypes) {
    return default_data[t].smina_name;
  }
  return "Unsupported";
}


///return type index and radius of a
std::pair<int,float> GninaIndexTyper::get_atom_type_index(OpenBabel::OBAtom* a) const {

  //this function is more convoluted than it needs to be for historical reasons
  //and a general fear of breaking backwards compatibility
  bool Hbonded = false;
  bool heteroBonded = false;


  FOR_NBORS_OF_ATOM(neigh, a){
    if (neigh->GetAtomicNum() == 1)
      Hbonded = true;
    else if (neigh->GetAtomicNum() != 6)
      heteroBonded = true; //hetero anything that is not hydrogen and not carbon
  }

  const char *element_name = GET_SYMBOL(a->GetAtomicNum());
  std::string ename(element_name);

  //massage the element name in some cases
  switch(a->GetAtomicNum()) {
  case 1:
    ename =  a->IsPolarHydrogen() ? "HD" : "H";
    break;
  case 6:
    if(a->IsAromatic()) ename = "A";
    break;
  case 7:
    if(a->IsHbondAcceptor()) ename = "NA";
    break;
  case 8:
    ename = "OA";
    break;
  case 16:
    if(a->IsHbondAcceptor()) ename = "SA";
    break;
  case 34:
    ename = "S"; //historically selenium is treated as sulfur  ¯\_(ツ)_/¯
    break;
  }

  //lookup type in data based on ename
  int ret = GenericMetal; //default catchall type
  for(int i = 0; i < NumTypes; i++) {
    if(data[i].adname == ename) {
      ret = i;
      break;
    }
  }
  //adjust based on bonding
  switch (ret) {
  case AliphaticCarbonXSHydrophobe: // C_C_C_H, //hydrophobic according to xscale
  case AliphaticCarbonXSNonHydrophobe: //C_C_C_P,
    ret = heteroBonded ?
            AliphaticCarbonXSNonHydrophobe : AliphaticCarbonXSHydrophobe;
    break;
  case AromaticCarbonXSHydrophobe: //C_A_C_H,
  case AromaticCarbonXSNonHydrophobe: //C_A_C_P,
    ret = heteroBonded ?
            AromaticCarbonXSNonHydrophobe : AromaticCarbonXSHydrophobe;
    break;
  case NitrogenXSDonor: //N_N_N_D,
  case Nitrogen: //N_N_N_P, no hydrogen bonding
    ret = Hbonded ? NitrogenXSDonor : Nitrogen;
    break;
  case NitrogenXSDonorAcceptor: //N_NA_N_DA, also an autodock acceptor
  case NitrogenXSAcceptor: //N_NA_N_A, also considered an acceptor by autodock
    ret = Hbonded ? NitrogenXSDonorAcceptor : NitrogenXSAcceptor;
    break;
  case OxygenXSDonor: //O_O_O_D,
  case Oxygen: //O_O_O_P,
    ret = Hbonded ? OxygenXSDonor : Oxygen;
    break;
  case OxygenXSDonorAcceptor: //O_OA_O_DA, also an autodock acceptor
  case OxygenXSAcceptor: //O_OA_O_A, also an autodock acceptor
    ret = Hbonded ? OxygenXSDonorAcceptor : OxygenXSAcceptor;
    break;
  }

  if(use_covalent) {
    return make_pair(ret, data[ret].covalent_radius);
  } else {
    return make_pair(ret, data[ret].xs_radius);
  }

}

//look up radius for passed type
pair<int,float> GninaIndexTyper::get_int_type(int t) const {
  int ret = GenericMetal;
  if(t < NumTypes) {
    ret = t;
  }
  if(use_covalent) {
    return make_pair(ret, data[ret].covalent_radius);
  } else {
    return make_pair(ret, data[ret].xs_radius);
  }
}

//return vector of string representations of types
std::vector<std::string> GninaIndexTyper::get_type_names() const {
  vector<string> ret; ret.reserve(NumTypes);
  for(unsigned i = 0; i < NumTypes; i++) {
    ret.push_back(data[i].smina_name);
  }
  return ret;
}

//return vector of atomic radii
std::vector<float> GninaIndexTyper::get_type_radii() const {
  vector<float> ret; ret.reserve(NumTypes);
  for(unsigned i = 0; i < NumTypes; i++) {
    if(use_covalent) {
      ret.push_back(data[i].covalent_radius);
    } else {
      ret.push_back(data[i].xs_radius);
    }
  }
  return ret;
}

/************** Element IndexTyper  ********************/

/// return number of types
unsigned ElementIndexTyper::num_types() const {
  return last_elem;
}

///return type index of a
std::pair<int,float> ElementIndexTyper::get_atom_type_index(OpenBabel::OBAtom* a) const {
  unsigned elem = a->GetAtomicNum();
  float radius = GET_COVALENT_RAD(elem);
  if(elem >= last_elem) elem = 0; //truncate
  return make_pair((int)elem,radius);
}

//return element with radius
std::pair<int,float> ElementIndexTyper::get_int_type(int elem) const {
  float radius = GET_COVALENT_RAD(elem);
  if(elem >= (int)last_elem) elem = 0; //truncate
  return make_pair((int)elem,radius);
}

//return vector of string representations of types
std::vector<std::string> ElementIndexTyper::get_type_names() const {
  vector<string> ret; ret.reserve(last_elem);
  ret.push_back("GenericAtom");
  for(unsigned i = 1; i < last_elem; i++) {
    ret.push_back(GET_NAME(i));
  }
  return ret;
}

//return vector of atomic radii
std::vector<float> ElementIndexTyper::get_type_radii() const {
  vector<float> ret; ret.reserve(last_elem+1);
  ret.push_back(0);
  for(unsigned i = 1; i < last_elem; i++) {
    float radius = GET_COVALENT_RAD(i);
    ret.push_back(radius);
  }
  return ret;
}


//safely set type_names from names, even if some are missing (use indices in that case)
void AtomIndexTyper::set_names(unsigned ntypes, std::vector<std::string>& type_names, const std::vector<std::string>& names) {
  type_names.clear();
  type_names.reserve(ntypes);
  for(unsigned i = 0; i < ntypes; i++) {
    if(i < names.size()) {
      type_names.push_back(names[i]);
    } else {
      type_names.push_back(boost::lexical_cast<string>(i));
    }
  }
}
CallbackIndexTyper::CallbackIndexTyper(AtomIndexTyperFunc f, unsigned ntypes, const std::vector<std::string>& names): callback(f) {
  //setup names
  set_names(ntypes, type_names, names);
}



/****************** GninaVectorTyper ******************/

vector<string> GninaVectorTyper::vtype_names { //this needs to match up with vtype enum
  "Hydrogen",
  "Carbon",
  "Nitrogen",
  "Oxygen",
  "Sulfur",
  "Phosphorus",
  "Fluorine",
  "Chlorine",
  "Bromine",
  "Iodine",
  "Magnesium",
  "Manganese",
  "Zinc",
  "Calcium",
  "Iron",
  "Boron",
  "GenericAtom",
  "AD_depth", //floating point
  "AD_solvation", //float
  "AD_volume", //float
  "XS_hydrophobe", //bool
  "XS_donor", //bool
  "XS_acceptor", //bool
  "AD_heteroatom",
  "Aromatic", //bool
  "OB_partialcharge"
};

/// return number of types
unsigned GninaVectorTyper::num_types() const {
  //there's the supported gnina elements plus the properties
  // ad_depth, ad_solvation, ad_volume, xs_hydrophobe, xs_donor, xs_acceptor, ad_heteroatom
  return NumTypes;
}

///return type index of a
float GninaVectorTyper::get_atom_type_vector(OpenBabel::OBAtom* a, std::vector<float>& typ) const {
  typ.assign(NumTypes, 0);
  auto t_r = ityper.get_atom_type_index(a);
  int t = t_r.first;
  float radius = t_r.second;

  int elemtyp = 0;
  switch (t) { //convert to element index
  case GninaIndexTyper::Hydrogen:
  case GninaIndexTyper::PolarHydrogen:
    elemtyp = Hydrogen;
    break;
  case GninaIndexTyper::AliphaticCarbonXSHydrophobe:
  case GninaIndexTyper::AliphaticCarbonXSNonHydrophobe:
  case GninaIndexTyper::AromaticCarbonXSHydrophobe:
  case GninaIndexTyper::AromaticCarbonXSNonHydrophobe:
    elemtyp = Carbon;
    break;

  case GninaIndexTyper::Nitrogen:
  case GninaIndexTyper::NitrogenXSDonor:
  case GninaIndexTyper::NitrogenXSDonorAcceptor:
  case GninaIndexTyper::NitrogenXSAcceptor:
    elemtyp = Nitrogen;
    break;
  case GninaIndexTyper::Oxygen:
  case GninaIndexTyper::OxygenXSDonor:
  case GninaIndexTyper::OxygenXSDonorAcceptor:
  case GninaIndexTyper::OxygenXSAcceptor:
    elemtyp = Oxygen;
    break;
  case GninaIndexTyper::Sulfur:
  case GninaIndexTyper::SulfurAcceptor:
    elemtyp = Sulfur;
    break;
  case GninaIndexTyper::Phosphorus:
    elemtyp = Phosphorus;
    break;
  case GninaIndexTyper::Fluorine:
    elemtyp = Fluorine;
    break;
  case GninaIndexTyper::Chlorine:
    elemtyp = Chlorine;
    break;
  case GninaIndexTyper::Bromine:
    elemtyp = Bromine;
    break;
  case GninaIndexTyper::Iodine:
    elemtyp = Iodine;
    break;
  case GninaIndexTyper::Magnesium:
    elemtyp = Magnesium;
    break;
  case GninaIndexTyper::Manganese:
    elemtyp = Manganese;
    break;
  case GninaIndexTyper::Zinc:
    elemtyp = Zinc;
    break;
  case GninaIndexTyper::Calcium:
    elemtyp = Calcium;
    break;
  case GninaIndexTyper::Iron:
    elemtyp = Iron;
    break;
  case GninaIndexTyper::GenericMetal:
    elemtyp = GenericAtom;
    break;
  case GninaIndexTyper::Boron:
    elemtyp = Boron;
    break;
  default:
    elemtyp = GenericAtom;
  }

  //set one-hot element
  typ[elemtyp] = 1.0;
  //set properties
  const GninaIndexTyper::info& info = ityper.get_info(t);
  typ[AD_depth] = info.ad_depth;
  typ[AD_solvation] = info.ad_solvation;
  typ[AD_volume] = info.ad_volume;
  typ[XS_hydrophobe] = info.xs_hydrophobe;
  typ[XS_donor] = info.xs_donor;
  typ[XS_acceptor] = info.xs_acceptor;
  typ[AD_heteroatom] = info.ad_heteroatom;
  typ[OB_partialcharge] = a->GetPartialCharge();
  typ[Aromatic] = a->IsAromatic();
  return radius;
}

///return radii of types
std::vector<float> GninaVectorTyper::get_vector_type_radii() const {
  std::vector<float> ret;
  for(unsigned i = 0; i < NumTypes; i++) {
    const GninaIndexTyper::info& info = ityper.get_info(i);
    ret.push_back(info.xs_radius);
  }
  return ret;
}

//return vector of string representations of types
std::vector<std::string> GninaVectorTyper::get_type_names() const {
  return vtype_names;
}


CallbackVectorTyper::CallbackVectorTyper(AtomVectorTyperFunc f, unsigned ntypes, const std::vector<std::string>& names): callback(f) {
  //setup names
  AtomIndexTyper::set_names(ntypes, type_names, names);
}


/*********** FileAtomMapper *****************/

/// read in map
void FileAtomMapper::setup(std::istream& in) {
  using namespace boost::algorithm;
  //first create reverse map from old names to old types
  unordered_map<string, int> old_name_to_old_type;
  for(unsigned i = 0, n = old_type_names.size(); i < n; i++) {
    old_name_to_old_type[old_type_names[i]] = i;
  }

  old_type_to_new_type.assign(old_type_names.size(), -1);
  new_type_names.clear();

  vector<vector<float> > radii; //indexed by new type (line), stores all radii of represnted types

  //each non blank line is a type
  string line;

  while (getline(in, line)) {
    trim(line);
    vector<string> types;
    split(types, line, is_any_of("\t \n"), boost::token_compress_on);
    if(types.size() > 0) {
      string new_type_name = join(types,"_");
      int ntype = new_type_names.size();
      new_type_names.push_back(new_type_name);

      //setup map
      for (unsigned i = 0, n = types.size(); i < n; i++) {
        const string& name = types[i];
        if(old_name_to_old_type.count(name)) {
          int oldt = old_name_to_old_type[name];
          old_type_to_new_type[oldt] = ntype;
        } else if (name.size() > 0){ //ignore consecutive delimiters
          string err("Invalid atom type ");
          err += name;
          throw invalid_argument(err);
        }
      }
    }
  }
}

FileAtomMapper::FileAtomMapper(const std::string& fname, const std::vector<std::string>& type_names): old_type_names(type_names) {
  ifstream in(fname.c_str());

  if(!in) {
    throw std::invalid_argument("Could not open " + fname);
  }

  setup(in);
}

/// return mapped type
int FileAtomMapper::get_new_type(unsigned origt) const {
  if(origt < old_type_to_new_type.size()) return old_type_to_new_type[origt];
  else return -1;
}

/*************** SubsetAtomMapper **********************/
SubsetAtomMapper::SubsetAtomMapper(const std::vector<int>& map,
    bool include_catchall, const std::vector<std::string>& old_names) {
  for(unsigned i = 0, n = map.size(); i < n; i++) {
    unsigned oldt = map[i];
    old2new[oldt] = i;
    if(oldt < old_names.size()) {
      new_type_names.push_back(old_names[oldt]);
    } else {
      new_type_names.push_back(boost::lexical_cast<string>(oldt));
    }
  }
  num_new_types = map.size();
  if(include_catchall) {
    default_type = map.size();
    num_new_types++;
    new_type_names.push_back("GenericAtom");
  }

}

///surjective mapping
SubsetAtomMapper::SubsetAtomMapper(const std::vector<std::vector<int> >& map,
    bool include_catchall, const std::vector<std::string>& old_names) {
  for(unsigned i = 0, n = map.size(); i < n; i++) {
    vector<string> names;
    for(unsigned j = 0, m = map[i].size(); j < m; j++) {
      unsigned oldt = map[i][j];
      old2new[oldt] = i;
      if(oldt < old_names.size()) {
        names.push_back(old_names[oldt]);
      } else {
        names.push_back(boost::lexical_cast<string>(oldt));
      }
    }
    string new_type_name = boost::algorithm::join(names,"_");
    new_type_names.push_back(new_type_name);
  }
  num_new_types = map.size();
  if(include_catchall) {
    default_type = map.size();
    num_new_types++;
    new_type_names.push_back("GenericAtom");
  }

}

/// return mapped type
int SubsetAtomMapper::get_new_type(unsigned origt) const {
  if(old2new.count(origt)) {
    return old2new.at(origt);
  }
  return default_type;
}

static stringstream recmap(R"(AliphaticCarbonXSHydrophobe 
AliphaticCarbonXSNonHydrophobe 
AromaticCarbonXSHydrophobe 
AromaticCarbonXSNonHydrophobe
Bromine Iodine Chlorine Fluorine
Nitrogen NitrogenXSAcceptor 
NitrogenXSDonor NitrogenXSDonorAcceptor
Oxygen OxygenXSAcceptor 
OxygenXSDonorAcceptor OxygenXSDonor
Sulfur SulfurAcceptor
Phosphorus 
Calcium
Zinc
GenericMetal Boron Manganese Magnesium Iron
)");
static stringstream ligmap(R"(AliphaticCarbonXSHydrophobe 
AliphaticCarbonXSNonHydrophobe 
AromaticCarbonXSHydrophobe 
AromaticCarbonXSNonHydrophobe
Bromine Iodine
Chlorine
Fluorine
Nitrogen NitrogenXSAcceptor 
NitrogenXSDonor NitrogenXSDonorAcceptor
Oxygen OxygenXSAcceptor 
OxygenXSDonorAcceptor OxygenXSDonor
Sulfur SulfurAcceptor
Phosphorus
GenericMetal Boron Manganese Magnesium Zinc Calcium Iron
)");
FileMappedGninaTyper defaultGninaReceptorTyper(recmap);
FileMappedGninaTyper defaultGninaLigandTyper(ligmap);

} /* namespace libmolgrid */
