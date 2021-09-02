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
#include <functional>

namespace libmolgrid {

/***** Base classes **********/

// Docstring_AtomTyper
/** \brief Base class for all atom typers */
class AtomTyper {
  public:
    AtomTyper() {}
    virtual ~AtomTyper() {}

    //why not make these abstract you ask? python bindings
    virtual unsigned num_types() const  { throw std::logic_error("Base class AtomTyper function called");  }

    virtual float get_atom_type_vector(OpenBabel::OBAtom *a, std::vector<float>& typ) const { throw std::logic_error("Unimplemented atom typing function called"); }
    virtual std::pair<int,float> get_atom_type_index(OpenBabel::OBAtom *a) const { throw std::logic_error("Unimplemented atom typing function called"); }
    virtual std::pair<int,float> get_int_type(int t) const { throw std::logic_error("Unimplemented atom typing function called"); }

    virtual std::vector<std::string> get_type_names() const { throw std::logic_error("Base class AtomTyper function called"); }
    virtual bool is_vector_typer() const { throw std::logic_error("Base class AtomTyper function called"); }
};

/** \brief Base class for generating numerical types along with atomic radius */
class AtomIndexTyper: public AtomTyper {
  public:
    AtomIndexTyper() {}
    virtual ~AtomIndexTyper() {}

    /// return number of types
    virtual unsigned num_types() const = 0;

    virtual float get_atom_type_vector(OpenBabel::OBAtom *a, std::vector<float>& typ) const final { throw std::logic_error("Vector typing called in index typer"); }

    ///return type index of a along with the apprioriate index
    virtual std::pair<int,float> get_atom_type_index(OpenBabel::OBAtom *a) const = 0;

    /// return type and radius given a precomputed type, the meaning of which
    /// is specific to the implementation
    virtual std::pair<int,float> get_int_type(int t) const = 0;

    ///return vector of string representations of types
    ///this isn't expected to be particularly efficient
    virtual std::vector<std::string> get_type_names() const = 0;

    ///if applicable to the typer, return the standard atomic radius of each type
    virtual std::vector<float> get_type_radii() const  {  return std::vector<float>(num_types(), 1.0); }

    static void set_names(unsigned ntypes, std::vector<std::string>& type_names, const std::vector<std::string>& names);

    virtual bool is_vector_typer() const { return false; };


};

/** \brief Base class for generating vector types */
class AtomVectorTyper: public AtomTyper {
  public:
    AtomVectorTyper() {}
    virtual ~AtomVectorTyper() {}

    /// return number of types
    virtual unsigned num_types() const = 0;

    virtual std::pair<int,float> get_atom_type_index(OpenBabel::OBAtom *a) const final { throw std::logic_error("Index typer called in vector typer"); }

    ///set vector type of atom a, return radius
    virtual float get_atom_type_vector(OpenBabel::OBAtom *a, std::vector<float>& typ) const = 0;

    ///return radii of types
    virtual std::vector<float> get_vector_type_radii() const { return std::vector<float>(num_types(), 1.0); }

    //return vector of string representations of types
    //this isn't expected to be particularly efficient
    virtual std::vector<std::string> get_type_names() const = 0;
    virtual bool is_vector_typer() const { return true; };


};

/** \brief Base class for mapping between type indices */
class AtomIndexTypeMapper {
  public:
    AtomIndexTypeMapper() {}
    virtual ~AtomIndexTypeMapper() {}

    /// return number of mapped types, zero if unknown (no mapping)
    virtual unsigned num_types() const { return 0; };

    /// return mapped type
    virtual int get_new_type(unsigned origt) const { return origt; }

    /// return vector of string representations of types
    virtual std::vector<std::string> get_type_names() const = 0;
};


// Docstring_GninaIndexTyper
/*********** Atom typers *****************/

/** \brief Calculate gnina types
 *
 * These are variants of AutoDock4 types. */
class GninaIndexTyper: public AtomIndexTyper {
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

    /** Information for an atom type.  This includes many legacy fields. */
    struct info
    {
      type sm;
      const char* smina_name; //this must be more than 2 chars long
      const char* adname;//this must be no longer than 2 chars
      unsigned anum;
      float ad_radius;
      float ad_depth;
      float ad_solvation;
      float ad_volume;
      float covalent_radius;
      float xs_radius;
      bool xs_hydrophobe;
      bool xs_donor;
      bool xs_acceptor;
      bool ad_heteroatom;
    };

  private:
    bool use_covalent = false;
    static const info default_data[NumTypes];
    const info *data = NULL; //data to use

  public:

    /** Create a gnina typer.
     * @param[in] usec  use the gnina determined covalent radius.
     */
    GninaIndexTyper(bool usec = false, const info *d = default_data): use_covalent(usec), data(d) {}
    virtual ~GninaIndexTyper() {}

    /// return number of types
    virtual unsigned num_types() const;

    ///return type index of a
    virtual std::pair<int,float> get_atom_type_index(OpenBabel::OBAtom* a) const;

    /// basically look up the radius of the given gnina type
    virtual std::pair<int,float> get_int_type(int t) const;

    ///return vector of string representations of types
    virtual std::vector<std::string> get_type_names() const;

    ///return atomic radius of each type
    virtual std::vector<float> get_type_radii() const;


    ///return gnina info for a given type
    const info& get_info(int t) const { return data[t]; }

    ///return name of default gnina type t
    static std::string gnina_type_name(int t);
};

// Docstring_ElementIndexTyper
/** \brief Calculate element types
 *
 * There are quite a few elements, so should probably run this through
 * an organic chem atom mapper that reduces to number of types.
 * The type id is the atomic number.  Any element with atomic number
 * greater than or equal to the specified max is assigned type zero.
 *  */
class ElementIndexTyper: public AtomIndexTyper {
    unsigned last_elem;
    const float default_radius = 1.6;
  public:
    ElementIndexTyper(unsigned maxe = 84): last_elem(maxe) {}
    virtual ~ElementIndexTyper() {}

    /// return number of types
    virtual unsigned num_types() const;

    ///return type index of a
    virtual std::pair<int,float> get_atom_type_index(OpenBabel::OBAtom* a) const;

    ///look up covalent radius of element or provide default
    virtual std::pair<int,float> get_int_type(int t) const;

    //return vector of string representations of types
    virtual std::vector<std::string> get_type_names() const;

    ///return atomic radius of each type, generic type is given zero radius
    virtual std::vector<float> get_type_radii() const;
};

// Docstring_NullIndexTyper
/** \brief Always return an invalid type
 *  */
class NullIndexTyper: public AtomIndexTyper {

  public:
    NullIndexTyper() {}
    virtual ~NullIndexTyper() {}

    /// return number of types
    virtual unsigned num_types() const { return 0; }

    ///return type index of a
    virtual std::pair<int,float> get_atom_type_index(OpenBabel::OBAtom* a) const { return std::make_pair(-1,0.0f); }

    ///look up covalent radius of element or provide default
    virtual std::pair<int,float> get_int_type(int t) const { return std::make_pair(-1,0.0f);}

    //return vector of string representations of types
    virtual std::vector<std::string> get_type_names() const { return std::vector<std::string>(); }

    ///return atomic radius of each type, generic type is given zero radius
    virtual std::vector<float> get_type_radii() const { return std::vector<float>(); }
};

/** \brief Use user-provided callback to do typing
 *  Must provide the number of types and their names.
 */
class CallbackIndexTyper: public AtomIndexTyper {
  public:
    using AtomIndexTyperFunc = std::function<std::pair<int,float>(OpenBabel::OBAtom* a)>;

  private:
    AtomIndexTyperFunc callback = nullptr;
    std::vector<std::string> type_names;
    const float default_radius = 1.6;

  public:

    /// iniitalize callbacktyper, if names are not provided, numerical names will be generated
    CallbackIndexTyper(AtomIndexTyperFunc f, unsigned ntypes, const std::vector<std::string>& names=std::vector<std::string>());

    /// return number of types
    virtual unsigned num_types() const { return type_names.size(); }

    ///return type index of a
    virtual std::pair<int,float> get_atom_type_index(OpenBabel::OBAtom* a) const {
      auto ret = callback(a);
      //don't allow out of range types
      if(ret.first >= (int)num_types()) ret.first = -1;
      return ret;
    }

    //callbacks are really only for obatom typing
    virtual std::pair<int,float> get_int_type(int t) const {
      if(t >= (int)num_types()) t = -1;
      return std::make_pair(t, default_radius);
    }

    //return vector of string representations of types
    virtual std::vector<std::string> get_type_names() const { return type_names; }

    virtual std::vector<float> get_type_radii() const  {  return std::vector<float>(num_types(), default_radius); }

};



/** \brief Wrap an atom typer with a mapper
 *
 */
template<class Mapper, class Typer>
class MappedAtomIndexTyper: public AtomIndexTyper {
  protected:
    Mapper mapper;
    Typer typer;

    std::vector<float> type_radii;
  public:
    MappedAtomIndexTyper(const Mapper& map, const Typer& typr): mapper(map), typer(typr) {
      unsigned oldN = typer.num_types();
      unsigned newN = mapper.num_types();

      std::vector< std::vector<float> > radii(newN);
      for(unsigned ot = 0; ot < oldN; ot++) {
        auto t_r = typer.get_int_type(ot);
        if(t_r.first >= 0) {
          unsigned nt = mapper.get_new_type(t_r.first);
          if(nt < radii.size()) {
            radii[nt].push_back(t_r.second);
          }
        }
      }

      type_radii.resize(newN);
      for(unsigned i = 0; i < newN; i++) {
        float sum = 0.0;
        for(unsigned j = 0, n = radii[i].size(); j < n; j++) {
          sum += radii[i][j];
        }
        type_radii[i] = sum/radii[i].size();
      }
    }

    virtual ~MappedAtomIndexTyper() {}

    /// return number of types
    virtual unsigned num_types() const {
      return mapper.num_types();
    }

    ///return type index of a
    virtual std::pair<int,float> get_atom_type_index(OpenBabel::OBAtom* a) const {
      auto res_rad = typer.get_atom_type_index(a);
      //remap the type
      int ret = mapper.get_new_type(res_rad.first);
      return std::make_pair(ret, res_rad.second);
    }

    //map the type
    virtual std::pair<int,float> get_int_type(int t) const {
      auto res_rad = typer.get_int_type(t);
      //remap the type
      int ret = mapper.get_new_type(res_rad.first);
      return std::make_pair(ret, res_rad.second);
    }

    //return vector of string representations of types
    virtual std::vector<std::string> get_type_names() const {
      return mapper.get_type_names();
    }

    ///radii are the average of the underlying mapped types
    virtual std::vector<float> get_type_radii() const  {  return type_radii; }

};

// Docstring_GninaVectorTyper
/** \brief Decompose gnina types into elements and properties.  Result is boolean.
 *
 * Hydrophobic, Aromatic, Donor, Acceptor
 *
 * These are variants of AutoDock4 types. */
class GninaVectorTyper: public AtomVectorTyper {
    GninaIndexTyper ityper;
    static std::vector<std::string> vtype_names;
  public:
    enum vtype {
      /* 0 */Hydrogen,
      /* 1 */Carbon,
      /* 2 */Nitrogen,
      /* 3 */Oxygen,
      /* 4 */Sulfur,
      /* 5 */Phosphorus,
      /* 6 */Fluorine,
      /* 7 */Chlorine,
      /* 8 */Bromine,
      /* 9 */Iodine,
      /* 10 */Magnesium,
      /* 11 */Manganese,
      /* 12 */Zinc,
      /* 13 */Calcium,
      /* 14 */Iron,
      /* 15 */Boron,
      /* 16 */GenericAtom,
      /* 17 */AD_depth, //floating point
      /* 18 */AD_solvation, //float
      /* 19 */AD_volume, //float
      /* 20 */XS_hydrophobe, //bool
      /* 21 */XS_donor, //bool
      /* 22 */XS_acceptor, //bool
      /* 23 */AD_heteroatom, //bool
      /* 24 */OB_partialcharge, //float
      /* 25 */Aromatic, //bool
      /* 26 */ NumTypes
    };

    GninaVectorTyper(const GninaIndexTyper& ityp = GninaIndexTyper()): ityper(ityp) {}
    virtual ~GninaVectorTyper() {}

    /// return number of types
    virtual unsigned num_types() const;

    ///return type index of a
    virtual float get_atom_type_vector(OpenBabel::OBAtom* a, std::vector<float>& typ) const;

    ///return radii of types
    virtual std::vector<float> get_vector_type_radii() const;

    //return vector of string representations of types
    virtual std::vector<std::string> get_type_names() const;
};


/** \brief Use user-provided callback to do vector typing
 *  Must provide the number of types and their names.
 */
class CallbackVectorTyper: public AtomVectorTyper {
  public:
    using AtomVectorTyperFunc = std::function<float (OpenBabel::OBAtom* a, std::vector<float>& )>;

  private:
    AtomVectorTyperFunc callback = nullptr;
    std::vector<std::string> type_names;

  public:

    /// iniitalize callbacktyper, if names are not provided, numerical names will be generated
    CallbackVectorTyper(AtomVectorTyperFunc f, unsigned ntypes, const std::vector<std::string>& names=std::vector<std::string>());

    /// return number of types
    virtual unsigned num_types() const { return type_names.size(); }

    ///set type vector and return radius for a
    virtual float get_atom_type_vector(OpenBabel::OBAtom* a, std::vector<float>& typ) const { return callback(a, typ); }

    //return vector of string representations of types
    virtual std::vector<std::string> get_type_names() const { return type_names; }
};


// Docstring_FileAtomMapper
/*********** Atom mappers *****************/

/** \brief Map atom types based on provided file.
 *
 * Each line for the provided file specifies a single type.
 * Types are specified using type names.
 * This class must be provided the type names properly indexed (should match get_type_names).
 */
class FileAtomMapper : public AtomIndexTypeMapper {
    std::vector<std::string> old_type_names;
    std::vector<int> old_type_to_new_type;
    std::vector<std::string> new_type_names;

    //setup map and new type names, assumes old_type_names is initialized
    void setup(std::istream& in);
  public:

    ///initialize from filename
    FileAtomMapper(const std::string& fname, const std::vector<std::string>& type_names);

    ///initialize from stream
    FileAtomMapper(std::istream& in, const std::vector<std::string>& type_names): old_type_names(type_names) {
      setup(in);
    }

    virtual ~FileAtomMapper() {}

    /// return number of mapped types, zero if unknown (no mapping)
    virtual unsigned num_types() const { return new_type_names.size(); }

    /// return mapped type
    virtual int get_new_type(unsigned origt) const;

    //return mapped type names
    virtual std::vector<std::string> get_type_names() const { return new_type_names; }

};

// Docstring_SubsetAtomMapper
/** \brief Map atom types onto a provided subset.
 */
class SubsetAtomMapper: public AtomIndexTypeMapper {
    std::unordered_map<int, int> old2new;
    std::vector<std::string> new_type_names;
    int default_type = -1; // if not in map
    unsigned num_new_types = 0;
  public:
    /// Indices of map are new types, values are the old types,
    /// if include_catchall is true, the last type will be the type
    /// returned for anything not in map (otherwise -1 is returned)
    SubsetAtomMapper(const std::vector<int>& map, bool include_catchall=true, const std::vector<std::string>& old_names = std::vector<std::string>());

    ///surjective mapping
    SubsetAtomMapper(const std::vector< std::vector<int> >& map, bool include_catchall=true, const std::vector<std::string>& old_names = std::vector<std::string>());

    /// return number of mapped types, zero if unknown (no mapping)
    virtual unsigned num_types() const { return num_new_types; }

    /// return mapped type
    virtual int get_new_type(unsigned origt) const;

    virtual std::vector<std::string> get_type_names() const { return new_type_names; }

};

// Docstring_SubsettedElementTyper
/// subsetting element types, derived class for convenient initialization
class SubsettedElementTyper: public MappedAtomIndexTyper<SubsetAtomMapper, ElementIndexTyper> {
  public:
    SubsettedElementTyper(const std::vector<int>& map, bool include_catchall=true, unsigned maxe = 84):
      SubsettedElementTyper(ElementIndexTyper(maxe), map, include_catchall) { //delegating constructor
    }

    SubsettedElementTyper(const ElementIndexTyper& etyper, const std::vector<int>& map, bool include_catchall=true):
      MappedAtomIndexTyper<SubsetAtomMapper, ElementIndexTyper>(
          SubsetAtomMapper(map,include_catchall,etyper.get_type_names()),etyper) {
    }

    SubsettedElementTyper(const std::vector< std::vector<int> >& map, bool include_catchall=true, unsigned maxe = 84):
      SubsettedElementTyper(ElementIndexTyper(maxe), map, include_catchall) { //delegating constructor
    }

    SubsettedElementTyper(const ElementIndexTyper& etyper, const std::vector< std::vector<int> >& map, bool include_catchall=true):
      MappedAtomIndexTyper<SubsetAtomMapper, ElementIndexTyper>(
          SubsetAtomMapper(map,include_catchall,etyper.get_type_names()),etyper) {
    }
};

/// subsetting gnina types, derived class for convenient initialization
class SubsettedGninaTyper: public MappedAtomIndexTyper<SubsetAtomMapper, GninaIndexTyper> {
  public:
    SubsettedGninaTyper(const std::vector<int>& map, bool include_catchall=true, bool usec = false):
      SubsettedGninaTyper(GninaIndexTyper(usec), map, include_catchall) { //delegating constructor to avoid multiple constructions of typer
    }

    SubsettedGninaTyper(const GninaIndexTyper& etyper, const std::vector<int>& map, bool include_catchall=true):
      MappedAtomIndexTyper<SubsetAtomMapper, GninaIndexTyper>(
          SubsetAtomMapper(map,include_catchall,etyper.get_type_names()),etyper) {
    }

    SubsettedGninaTyper(const std::vector< std::vector<int> >& map, bool include_catchall=true, bool usec = false):
      SubsettedGninaTyper(GninaIndexTyper(usec), map, include_catchall) { //delegating constructor
    }

    SubsettedGninaTyper(const GninaIndexTyper& etyper, const std::vector< std::vector<int> >& map, bool include_catchall=true):
      MappedAtomIndexTyper<SubsetAtomMapper, GninaIndexTyper>(
          SubsetAtomMapper(map,include_catchall,etyper.get_type_names()),etyper) {
    }
};

/// file mapping element types, derived class for convenient initialization
class FileMappedElementTyper: public MappedAtomIndexTyper<FileAtomMapper, ElementIndexTyper> {
  public:
    FileMappedElementTyper(const std::string& fname, unsigned maxe = 84):
      FileMappedElementTyper(ElementIndexTyper(maxe), fname) { //delegating constructor
    }

    FileMappedElementTyper(const ElementIndexTyper& etyper, const std::string& fname):
      MappedAtomIndexTyper<FileAtomMapper, ElementIndexTyper>(
          FileAtomMapper(fname,etyper.get_type_names()),etyper) {
    }

    FileMappedElementTyper(std::istream& i, unsigned maxe = 84):
      FileMappedElementTyper(ElementIndexTyper(maxe), i) { //delegating constructor
    }

    FileMappedElementTyper(const ElementIndexTyper& etyper, std::istream& i):
      MappedAtomIndexTyper<FileAtomMapper, ElementIndexTyper>(
          FileAtomMapper(i,etyper.get_type_names()),etyper) {
    }
};

/// file mapping element types, derived class for convenient initialization
class FileMappedGninaTyper: public MappedAtomIndexTyper<FileAtomMapper, GninaIndexTyper> {
  public:
    FileMappedGninaTyper(const std::string& fname, bool usec = false):
      FileMappedGninaTyper(GninaIndexTyper(usec), fname) { //delegating constructor
    }

    FileMappedGninaTyper(const GninaIndexTyper& etyper, const std::string& fname):
      MappedAtomIndexTyper<FileAtomMapper, GninaIndexTyper>(
          FileAtomMapper(fname,etyper.get_type_names()),etyper) {
    }

    FileMappedGninaTyper(std::istream& i, bool usec = false):
      FileMappedGninaTyper(GninaIndexTyper(usec), i) { //delegating constructor
    }

    FileMappedGninaTyper(const GninaIndexTyper& etyper, std::istream& i):
      MappedAtomIndexTyper<FileAtomMapper, GninaIndexTyper>(
          FileAtomMapper(i,etyper.get_type_names()),etyper) {
    }
};

/** \brief default types for receptor
AliphaticCarbonXSHydrophobe
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
 */
extern FileMappedGninaTyper defaultGninaReceptorTyper;

/** \brief default types for ligand
AliphaticCarbonXSHydrophobe
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
 */
extern FileMappedGninaTyper defaultGninaLigandTyper;

} /* namespace libmolgrid */

#endif /* ATOMTYPER_H_ */
