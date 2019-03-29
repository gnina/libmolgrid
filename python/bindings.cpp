/*
 * bindings.cpp
 *
 *  Python bindings for libmolgrid
 */

#include "bindings.h"
#include "quaternion.h"
#include "transform.h"
#include "atom_typer.h"
#include "example_provider.h"
#include "grid_maker.h"

using namespace boost::python;
using namespace libmolgrid;


/// indicate how MGrid shoudl be automatically converted in python bindings
bool python_gpu_enabled = true;

#include "boost/python.hpp"
#include "boost/python/detail/api_placeholder.hpp"

//#include "bindings_grids.cpp"

//https://wiki.python.org/moin/boost.python/HowTo#A.22Raw.22_constructor
namespace boost { namespace python {

namespace detail {

  template <class F>
  struct raw_constructor_dispatcher
  {
      raw_constructor_dispatcher(F f)
     : f(make_constructor(f)) {}

      PyObject* operator()(PyObject* args, PyObject* keywords)
      {
          borrowed_reference_t* ra = borrowed_reference(args);
          object a(ra);
          return incref(
              object(
                  f(
                      object(a[0])
                    , object(a.slice(1, len(a)))
                    , keywords ? dict(borrowed_reference(keywords)) : dict()
                  )
              ).ptr()
          );
      }

   private:
      object f;
  };

} // namespace detail

template <class F>
object raw_constructor(F f, std::size_t min_args = 0)
{
    return detail::make_raw_function(
        objects::py_function(
            detail::raw_constructor_dispatcher<F>(f)
          , mpl::vector2<void, object>()
          , min_args+1
          , (std::numeric_limits<unsigned>::max)()
        )
    );
}

}} // namespace boost::python


BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(Transform_forward_overloads, Transform::forward, 2, 3)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(Transform_backward_overloads, Transform::backward, 2, 3)


struct PySwigObject {
    PyObject_HEAD
    void * ptr;
    void *ty;
    int own;
};

//convert c++ obatom pointer to python object
object obatom_to_object(OpenBabel::OBAtom* a) {
  //this uses a complete hack of constructing a dummy atom and then
  //swapping out the underlying pointer
  object ob = import("openbabel");
  object atom = ob.attr("OBAtom")();

  PyObject *obj = atom.ptr();
  //first we need to get the this attribute from the Python Object
  if (!PyObject_HasAttrString(obj, "this"))
      return atom;

  PyObject* thisAttr = PyObject_GetAttrString(obj, "this");
  if (thisAttr == nullptr)
      return atom;

  //This Python Object is a SWIG Wrapper and contains our pointer
  OpenBabel::OBAtom* oldptr = (OpenBabel::OBAtom*)((PySwigObject*)thisAttr)->ptr;
  ((PySwigObject*)thisAttr)->ptr = a;
  ((PySwigObject*)thisAttr)->own = 0; //the molecule owns the memory
  delete oldptr;

  PyObject_SetAttrString(obj, "this", thisAttr);
  return atom;
}

void* extract_swig_wrapped_pointer(PyObject* obj)
{
    //first we need to get the this attribute from the Python Object
    if (!PyObject_HasAttrString(obj, "this"))
        return NULL;

    PyObject* thisAttr = PyObject_GetAttrString(obj, "this");
    if (thisAttr == nullptr)
        return nullptr;

    //This Python Object is a SWIG Wrapper and contains our pointer
    void* pointer = ((PySwigObject*)thisAttr)->ptr;
    Py_DECREF(thisAttr);
    return pointer;
}

//auto-unwrap obatom from pybel atom
void* extract_pybel_atom(PyObject *obj) {

  if (!PyObject_HasAttrString(obj, "OBAtom"))
    return nullptr;

  PyObject* obatom = PyObject_GetAttrString(obj, "OBAtom");
  if(obatom == nullptr)
    return nullptr;

  return extract_swig_wrapped_pointer(obatom);
}

//auto-unwrap obmol from pybel molecule
void* extract_pybel_molecule(PyObject *obj) {

  if (!PyObject_HasAttrString(obj, "OBMol"))
    return nullptr;

  PyObject* obatom = PyObject_GetAttrString(obj, "OBMol");
  if(obatom == nullptr)
    return nullptr;

  return extract_swig_wrapped_pointer(obatom);
}

// convert a python list to a uniformly typed vector
template<typename T>
std::vector<T> list_to_vec(list l) {
  unsigned n = len(l);
  std::vector<T> ret; ret.reserve(n);
  for(unsigned i = 0; i < n; i++) {
    ret.push_back(extract<T>(l[i]));
  }
  return ret;
}

//convert a list of lists to a vector of uniformly typed vectors, sublists can be single elements
template<typename T>
std::vector< std::vector<T> > listlist_to_vecvec(list l) {
  unsigned n = len(l);
  std::vector< std::vector<T> > ret; ret.reserve(n);
  for(unsigned i = 0; i < n; i++) {
    extract<T> singleton(l[i]);
    if(singleton.check()) {
      ret.push_back(std::vector<T>(1, singleton()));
    } else {
      ret.push_back(list_to_vec<T>(extract<list>(l[i])));
    }
  }
  return ret;
}

/** \brief Callback to python for index type
 */
class PythonCallbackIndexTyper: public CallbackIndexTyper {

    boost::python::object callback;

  public:

    /// iniitalize callbacktyper, if names are not provided, numerical names will be generated
    PythonCallbackIndexTyper(boost::python::object c, unsigned ntypes, list names):
          CallbackIndexTyper([this](OpenBabel::OBAtom* a) -> std::pair<int,float> {
      return extract< std::pair<int,float> >(callback(obatom_to_object(a)));
    }, ntypes, list_to_vec<std::string>(names)), callback(c) {
    }

    ///call callback
    // note I'm unwrapping and rewrapping the obatom in python mostly to test the code
    std::pair<int,float> get_atom_type_index(object a) const {
      OpenBabel::OBAtom *atom = (OpenBabel::OBAtom *)extract_swig_wrapped_pointer(a.ptr());
      if(atom)
        return CallbackIndexTyper::get_atom_type_index(atom);
      else
        throw std::invalid_argument("Need OBAtom");
    }
};

/** \brief Callback to python for vector type
 * The call back function should return a tuple ([type vector], radius) for the atom
 */
class PythonCallbackVectorTyper: public CallbackVectorTyper {

    boost::python::object callback;

  public:

    /// iniitalize callbacktyper, if names are not provided, numerical names will be generated
    PythonCallbackVectorTyper(boost::python::object c, unsigned ntypes, list lnames):
      CallbackVectorTyper([this](OpenBabel::OBAtom* a, std::vector<float>& typ) {
          object o = callback(obatom_to_object(a));
          tuple t(o);
          list vec(t[0]);
          float r = extract<float>(t[1]);
          typ = list_to_vec<float>(vec);
          return r;
          },
          ntypes, list_to_vec<std::string>(lnames)), callback(c) {
    }

    ///call callback - for python return vector by reference
    virtual tuple get_atom_type_vector(object a) const {
      OpenBabel::OBAtom *atom = (OpenBabel::OBAtom *)extract_swig_wrapped_pointer(a.ptr());
      if(atom) {
       std::vector<float> typ;
       float r = CallbackVectorTyper::get_atom_type_vector(atom, typ);
       return make_tuple(list(typ), r);
      } else {
       throw std::invalid_argument("Need OBAtom");
      }
    }
};

template<typename T1, typename T2>
struct PairToPythonConverter {
    static PyObject* convert(const std::pair<T1, T2>& pair)
    {
        return incref(make_tuple(pair.first, pair.second).ptr());
    }
};

template<typename T1, typename T2>
struct PythonToPairConverter {
    PythonToPairConverter()
    {
        converter::registry::push_back(&convertible, &construct, type_id<std::pair<T1, T2> >());
    }
    static void* convertible(PyObject* obj)
    {
        if (!PyTuple_CheckExact(obj)) return 0;
        if (PyTuple_Size(obj) != 2) return 0;
        return obj;
    }
    static void construct(PyObject* obj, converter::rvalue_from_python_stage1_data* data)
    {
        tuple tuple(borrowed(obj));
        void* storage = ((converter::rvalue_from_python_storage<std::pair<T1, T2> >*) data)->storage.bytes;
        new (storage) std::pair<T1, T2>(extract<T1>(tuple[0]), extract<T2>(tuple[1]));
        data->convertible = storage;
    }
};

template<typename T1, typename T2>
struct py_pair {
    to_python_converter<std::pair<T1, T2>, PairToPythonConverter<T1, T2> > toPy;
    PythonToPairConverter<T1, T2> fromPy;
};


//conversion of tuple to float3
struct PythonToFloat3Converter {

    PythonToFloat3Converter()
    {
        converter::registry::push_back(&convertible, &construct, type_id<float3>());

    }
    static void* convertible(PyObject* obj)
    {
        if (!PyTuple_CheckExact(obj)) return 0;
        if (PyTuple_Size(obj) != 3) return 0;
        return obj;
    }
    static void construct(PyObject* obj, converter::rvalue_from_python_stage1_data* data)
    {
        tuple tuple(borrowed(obj));
        void* storage = ((converter::rvalue_from_python_storage<float3 >*) data)->storage.bytes;
        float x = extract<float>(tuple[0]);
        float y = extract<float>(tuple[1]);
        float z = extract<float>(tuple[2]);
        new (storage) float3{x,y,z};
        data->convertible = storage;
    }
};


//return true if list is uniformly typed to T
template<typename T>
bool list_is_vec(list l) {
  unsigned n = len(l);
  for(unsigned i = 0; i < n; i++) {
    extract<T> e(l[i]);
    if(!e.check()) return false;
  }
  return true;
}

static void set_settings_form_kwargs(dict kwargs, ExampleProviderSettings& settings) {
  //extract any settings
  using namespace std;
  boost::python::list keys = kwargs.keys();
  for(unsigned i = 0, n = len(keys); i < n; i++) {
    object k = keys[i];
    string name = extract<string>(k);
    //preprocessor macros are beautifully ugly
#undef EXSET
#define EXSET(TYPE, NAME, DEFAULT, DOC) if(name == #NAME) { settings.NAME = extract<TYPE>(kwargs[k]); } else
    MAKE_SETTINGS() {throw invalid_argument("Unknown keyword argument "+name);}
  }
}

//raw constructor for example provider, args should be typers and kwargs
//settings in exampleprovidersettings
std::shared_ptr<ExampleProvider> create_ex_provider(tuple args, dict kwargs) {
  using namespace std;
  ExampleProviderSettings settings;

  //hard code some number of typers since this needs to be done statically
  int N = len(args);

  //first positional argument can be a settings object
  if(N > 0) {
    extract<ExampleProviderSettings> maybe_settings(args[0]);
    if (maybe_settings.check()) {
      settings = maybe_settings();
      args = boost::python::tuple(args.slice(1,_)); //remove
      N--;
    }
  }

  //kwargs take precedence over default/object
  set_settings_form_kwargs(kwargs, settings);

  vector<shared_ptr<AtomTyper> > typers;
  for(int i = 0; i < N; i++) {
    typers.push_back(extract<shared_ptr<AtomTyper> >(args[i]));
  }

  if(N == 0)
    return std::make_shared<ExampleProvider>(settings);
  else
    return std::make_shared<ExampleProvider>(settings, typers);

}

BOOST_PYTHON_MODULE(molgrid)
{
  Py_Initialize();
  bool numpy_supported = init_numpy();

  def("set_random_seed", +[](long s) {random_engine.seed(s);}); //set random seed
  def("get_gpu_enabled", +[]()->bool {return python_gpu_enabled;},
      "Get if generated grids are on GPU by default.");
  def("set_gpu_enabled", +[](bool val) {python_gpu_enabled = val;},
      "Set if generated grids should be on GPU by default.");
  def("tofloatptr", +[](long val) { return Pointer<float>((float*)val);}, "Return integer as float *");
  def("todoubleptr", +[](long val) { return Pointer<double>((double*)val);}, "Return integer as double *");

  //type converters
  py_pair<int, float>();
  py_pair<std::vector<float>, float>();
  py_pair<list, float>();
  PythonToFloat3Converter();

// Grids

//Grid bindings
#undef MAKE_GRID_TN
#define MAKE_GRID_TN(N, TYPE, NAME) define_grid<TYPE,NTYPES(N,unsigned)>(NAME, numpy_supported);

// MGrid bindings
#undef MAKE_MGRID_TN
#define MAKE_MGRID_TN(N, TYPE, NAME) define_mgrid<TYPE,NTYPES(N,unsigned)>(NAME);

MAKE_ALL_GRIDS()

//vector utility types
  class_<std::vector<size_t> >("SizeVec")
      .def(vector_indexing_suite<std::vector<size_t> >());
  class_<std::vector<std::string> >("StringVec")
      .def(vector_indexing_suite<std::vector<std::string> >());
  class_<std::vector<float> >("FloatVec")
      .def(vector_indexing_suite<std::vector<float> >());
  class_<std::vector<CoordinateSet> >("CoordinateSetVec")
      .def(vector_indexing_suite<std::vector<CoordinateSet> >());
  class_<std::vector<Example> >("ExampleVec")
      .def(vector_indexing_suite<std::vector<Example> >());

  class_<Pointer<float> >("FloatPtr", no_init);
  class_<Pointer<double> >("DoublePtr", no_init);

  class_<float3>("float3", no_init)
      .def("__init__",
      make_constructor(
          +[](float x, float y, float z) {return std::make_shared<float3>(make_float3(x,y,z));}))
      .def("__init__",
          make_constructor(+[](tuple t) {
          float x = extract<float>(t[0]);
          float y = extract<float>(t[1]);
          float z = extract<float>(t[2]);
          return std::make_shared<float3>(make_float3(x,y,z));
      }))
      .def("__getitem__",
          +[](const float3& f, size_t i) { //enable conversion to iterable taking types like tuple
        if(i == 0) return f.x;
        if(i == 1) return f.y;
        if(i == 2) return f.z;
        throw std::out_of_range("float3");
      })
      .def_readwrite("x", &float3::x)
      .def_readwrite("y", &float3::y)
      .def_readwrite("z", &float3::z);

// Quaternion - I lament the need for a custom quaternion class, yet here we are
  class_<Quaternion>("Quaternion")
      .def(init<float, float, float, float>())
      .def("R_component_1", &Quaternion::R_component_1)
      .def("R_component_2", &Quaternion::R_component_2)
      .def("R_component_3", &Quaternion::R_component_3)
      .def("R_component_4", &Quaternion::R_component_4)
      .def("real", &Quaternion::real)
      .def("conj", &Quaternion::conj)
      .def("norm", &Quaternion::norm)
      .def("rotate", &Quaternion::rotate, (arg("x"),"y","z"))
      .def("transform", &Quaternion::transform)
      .def("inverse", &Quaternion::inverse)
      .def(self * self)
      .def(self *= self)
      .def(self / self)
      .def(self /= self);

// Transform

  class_<Transform>("Transform")
      .def(init<Quaternion>())
      .def(init<Quaternion, float3>())
      .def(init<Quaternion, float3, float3>())
  .def(init<float3, float, bool>((arg("center"),arg("random_translate")=0.0,arg("random_rotation")=false))) //center, translate, rotate
  .def("quaternion", &Transform::quaternion, return_value_policy<copy_const_reference>())
  .def("rotation_center", &Transform::rotation_center)
  .def("translation", &Transform::translation)
  //non-const references need to be passed by value, so wrap
  .def("forward", +[](Transform& self, const Grid2f& in, Grid2f out, bool dotranslate) {self.forward(in,out,dotranslate);},
      Transform_forward_overloads((arg("in"), arg("out"), arg("dotranslate")=true)))
  .def("forward",  +[](Transform& self, const Grid2fCUDA& in, Grid2fCUDA out, bool dotranslate) {self.forward(in,out,dotranslate);},
      Transform_forward_overloads((arg("in"), arg("out"), arg("dotranslate")=true)))
  .def("forward", static_cast<void (Transform::*)(const CoordinateSet&, CoordinateSet&, bool) const>(&Transform::forward),
      Transform_forward_overloads((arg("in"), arg("out"), arg("dotranslate")=true)))
  .def("forward", static_cast<void (Transform::*)(const Example&, Example&, bool) const>(&Transform::forward),
          Transform_forward_overloads((arg("in"), arg("out"), arg("dotranslate")=true)))
  .def("backward", +[](Transform& self, const Grid2f& in, Grid2f out, bool dotranslate) {self.backward(in,out,dotranslate);},
      Transform_backward_overloads((arg("in"), arg("out"), arg("dotranslate")=true)))
  .def("backward",+[](Transform& self, const Grid2fCUDA& in, Grid2fCUDA out, bool dotranslate) {self.backward(in,out,dotranslate);},
       Transform_backward_overloads((arg("in"), arg("out"), arg("dotranslate")=true)));

//Atom typing
  converter::registry::insert(&extract_swig_wrapped_pointer, type_id<OpenBabel::OBAtom>());
  converter::registry::insert(&extract_pybel_atom, type_id<OpenBabel::OBAtom>());
  converter::registry::insert(&extract_swig_wrapped_pointer, type_id<OpenBabel::OBMol>());
  converter::registry::insert(&extract_pybel_molecule, type_id<OpenBabel::OBMol>());

  class_<AtomTyper>("AtomTyper", no_init);

  class_<GninaIndexTyper, bases<AtomTyper> >("GninaIndexTyper")
      .def(init<bool>())
      .def("num_types", &GninaIndexTyper::num_types)
      .def("get_atom_type_index", &GninaIndexTyper::get_atom_type_index)
      .def("get_type_names",&GninaIndexTyper::get_type_names);

  class_<ElementIndexTyper, bases<AtomTyper> >("ElementIndexTyper")
      .def(init<int>())
      .def("num_types", &ElementIndexTyper::num_types)
      .def("get_atom_type_index", &ElementIndexTyper::get_atom_type_index)
      .def("get_type_names",&ElementIndexTyper::get_type_names);

  class_<PythonCallbackIndexTyper, bases<AtomTyper> >("PythonCallbackIndexTyper",
      init<object, unsigned, list>(
          (arg("func"), arg("num_types"), arg("names") = list() ) ))
      .def("num_types", &PythonCallbackIndexTyper::num_types)
      .def("get_atom_type_index", &PythonCallbackIndexTyper::get_atom_type_index)
      .def("get_type_names",&PythonCallbackIndexTyper::get_type_names);

  class_<GninaVectorTyper, bases<AtomTyper> >("GninaVectorTyper")
      .def("num_types", &GninaVectorTyper::num_types)
      .def("get_atom_type_vector", +[](const GninaVectorTyper& typer, OpenBabel::OBAtom* a) {
        std::vector<float> typs;
        float r = typer.get_atom_type_vector(a, typs);
        auto ltyps = list(typs);
        return std::make_pair(ltyps,r);
        })
      .def("get_type_names",&GninaVectorTyper::get_type_names);

  class_<PythonCallbackVectorTyper, bases<AtomTyper> >("PythonCallbackVectorTyper",
      init<object, unsigned, list>(
          (arg("func"), arg("num_types"), arg("names") = list() ) ))
      .def("num_types", &PythonCallbackVectorTyper::num_types)
      .def("get_atom_type_vector", &PythonCallbackVectorTyper::get_atom_type_vector)
      .def("get_type_names",&PythonCallbackVectorTyper::get_type_names);

  class_<FileAtomMapper, bases<AtomTyper> >("FileAtomMapper", init<const std::string&, const std::vector<std::string> >())
      .def("num_types", &FileAtomMapper::num_types)
      .def("get_new_type", &FileAtomMapper::get_new_type)
      .def("get_type_names",&FileAtomMapper::get_type_names);

  class_<SubsetAtomMapper, bases<AtomTyper> >("SubsetAtomMapper", init<const std::vector<int>&, bool>())
      .def(init<const std::vector< std::vector<int> >&, bool>())
      .def(init<std::vector<int>&, bool, std::vector< std::string> >())
      .def(init<std::vector< std::vector<int> >&, bool, const std::vector< std::string>& >())
      .def("__init__",make_constructor(
          +[](list l, bool catchall, const std::vector< std::string>& old_names) {
          if(list_is_vec<int>(l)) {
            return std::make_shared<SubsetAtomMapper>(list_to_vec<int>(l),catchall,old_names);
          } else { //assume list of lists
            return std::make_shared<SubsetAtomMapper>(listlist_to_vecvec<int>(l),catchall,old_names);
          }
        }, default_call_policies(),
        (arg("map"), arg("catchall") = true, arg("old_names") = std::vector<std::string>())))
      .def("num_types", &SubsetAtomMapper::num_types)
      .def("get_new_type", &SubsetAtomMapper::get_new_type)
      .def("get_type_names",&SubsetAtomMapper::get_type_names);

  class_<SubsettedElementTyper, bases<AtomTyper> >("SubsettedElementTyper", no_init)
          .def("__init__",make_constructor(
              +[](list l, bool catchall, unsigned maxe) {
              if(list_is_vec<int>(l)) {
                return std::make_shared<SubsettedElementTyper>(list_to_vec<int>(l),catchall,maxe);
              } else { //assume list of lists
                return std::make_shared<SubsettedElementTyper>(listlist_to_vecvec<int>(l),catchall,maxe);
              }
            }, default_call_policies(),
            (arg("map"), arg("catchall") = true, arg("maxe") = 84U)))
          .def("num_types", &SubsettedElementTyper::num_types)
          .def("get_atom_type_index", &SubsettedElementTyper::get_atom_type_index)
          .def("get_type_names",&SubsettedElementTyper::get_type_names);

  class_<SubsettedGninaTyper, bases<AtomTyper> >("SubsettedGninaTyper", no_init)
         .def("__init__",make_constructor(
                  +[](list l, bool catchall, bool usec) {
                  if(list_is_vec<int>(l)) {
                    return std::make_shared<SubsettedGninaTyper>(list_to_vec<int>(l),catchall,usec);
                  } else { //assume list of lists
                    return std::make_shared<SubsettedGninaTyper>(listlist_to_vecvec<int>(l),catchall,usec);
                  }
                }, default_call_policies(),
                (arg("map"), arg("catchall") = true, arg("use_covalent_radius") = false)))
          .def("num_types", &SubsettedGninaTyper::num_types)
          .def("get_atom_type_index", &SubsettedGninaTyper::get_atom_type_index)
          .def("get_type_names",&SubsettedGninaTyper::get_type_names);

  class_<FileMappedGninaTyper, bases<AtomTyper> >("FileMappedGninaTyper",
          init<const std::string&, bool>((arg("fname"), arg("use_covalent_radius")=false)))
              //todo, add init for file stream inputs if we every want it
          .def("num_types", &FileMappedGninaTyper::num_types)
          .def("get_atom_type_index", &FileMappedGninaTyper::get_atom_type_index)
          .def("get_type_names",&FileMappedGninaTyper::get_type_names);

  class_<FileMappedElementTyper, bases<AtomTyper> >("FileMappedElementTyper",
          init<const std::string&, unsigned>((arg("fname"), arg("maxe")=84)))
          .def("num_types", &FileMappedElementTyper::num_types)
          .def("get_atom_type_index", &FileMappedElementTyper::get_atom_type_index)
          .def("get_type_names",&FileMappedElementTyper::get_type_names);

  scope().attr("defaultGninaLigandTyper") = defaultGninaLigandTyper;
  scope().attr("defaultGninaReceptorTyper") = defaultGninaReceptorTyper;


  //molecular data (example providing)
  class_<CoordinateSet>("CoordinateSet")
      .def(init<OpenBabel::OBMol*, const AtomTyper&>())
      .def(init<OpenBabel::OBMol*>())
      .def(init<const Grid2f&, const Grid1f&, const Grid1f&, unsigned>())
      .def(init<const Grid2fCUDA&, const Grid1fCUDA&, const Grid1fCUDA&, unsigned>())
      .def(init<const Grid2f&, const Grid2f&, const Grid1f&>())
      .def(init<const Grid2fCUDA&, const Grid2fCUDA&, const Grid1fCUDA&>())
      .def("has_indexed_types", &CoordinateSet::has_indexed_types)
      .def("has_vector_types", &CoordinateSet::has_vector_types)
      .def("make_vector_types", &CoordinateSet::make_vector_types)
      .def("size", &CoordinateSet::size)
      .def("num_types", &CoordinateSet::num_types)
      .def("center", &CoordinateSet::center)
      .def("togpu", &CoordinateSet::togpu, "set memory affinity to GPU")
      .def("tocpu", &CoordinateSet::tocpu, "set memory affinity to CPU")
      .def_readwrite("coord", &CoordinateSet::coord)
      .def_readwrite("type_index", &CoordinateSet::type_index)
      .def_readwrite("type_vector", &CoordinateSet::type_vector)
      .def_readwrite("radius", &CoordinateSet::radius)
      .def_readwrite("max_type", &CoordinateSet::max_type)
      .def_readonly("src", &CoordinateSet::src);


  //mostly exposing this for documentation purposes
#undef EXSET
#define EXSET(TYPE, NAME, DEFAULT, DOC) .def_readwrite(#NAME, &ExampleProviderSettings::NAME, DOC)

  class_<ExampleProviderSettings>("ExampleProviderSettings")
      MAKE_SETTINGS();

  class_<Example>("Example")
    .def("coordinate_size", &Example::coordinate_size)
    .def("type_size", &Example::type_size, (arg("unique_index_type")=true))
    .def("merge_coordinates", static_cast<CoordinateSet (Example::*)(bool)>(&Example::merge_coordinates), (arg("unique_index_types") = true))
    .def("merge_coordinates", static_cast<void (Example::*)(Grid2f&, Grid1f&, Grid1f&, bool)>(&Example::merge_coordinates), (arg("coord"), "type_index", "radius", arg("unique_index_types")=true))
    .def("merge_coordinates", static_cast<void (Example::*)(Grid2f&, Grid2f&, Grid1f&, bool)>(&Example::merge_coordinates), (arg("coord"), "type_vector", "radius", arg("unique_index_types")=true))
    .def("togpu", &Example::togpu, "set memory affinity to GPU")
    .def("tocpu", &Example::tocpu, "set memory affinity to CPU")
    .def_readwrite("coord_sets",&Example::sets)
    .def_readwrite("labels",&Example::labels);

  //there is quite a lot of functionality in the C++ api for example providers, but keep it simple in python for now
  class_<ExampleProvider>("ExampleProvider")
      .def("__init__", raw_constructor(&create_ex_provider,0),"Construct an ExampleProvider using an ExampleSettings object "
          "and the desired AtomTypers for each molecule.  Alternatively, specify individual settings using keyword arguments")
      .def("populate",
          static_cast<void (ExampleProvider::*)(const std::string&, int, bool)>(&ExampleProvider::populate),
          (arg("file_name"), arg("num_labels")=-1, arg("has_group")=false))
      .def("populate", +[](ExampleProvider& self, list l, int num_labels, bool has_group) {
            if(list_is_vec<std::string>(l)) {
                self.populate(list_to_vec<std::string>(l), num_labels, has_group);
              } else {
                throw std::invalid_argument("Need list of file names for ExampleProvider");
              }
          },
          (arg("file_name"), arg("num_labels")=-1, arg("has_group")=false))
      .def("type_size", &ExampleProvider::type_size)
      .def("next", static_cast<Example (ExampleProvider::*)()>(&ExampleProvider::next))
      .def("next_batch", static_cast< std::vector<Example> (ExampleProvider::*)(unsigned)>(&ExampleProvider::next_batch),
          (arg("batch_size")));


  //grid maker
  class_<GridMaker>("GridMaker",
      init<float, float, float, bool>((arg("resolution")=0.5, arg("dimension")=23.5, arg("radius_multiple")=1.5, arg("binary")=false)))
      .def("spatial_grid_dimensions", +[](GridMaker& self) { float3 dims = self.getGridDims(); return make_tuple(int(dims.x),int(dims.y),int(dims.z));})
      .def("grid_dimensions", +[](GridMaker& self, int ntypes) { float3 dims = self.getGridDims(); return make_tuple(ntypes,int(dims.x),int(dims.y),int(dims.z));})
      //grids need to be passed by value
      .def("forward", +[](GridMaker& self, float3 center, const CoordinateSet& c, Grid<float, 4, false> g){ self.forward(center, c, g); })
      .def("forward", +[](GridMaker& self, float3 center, const CoordinateSet& c, Grid<float, 4, true> g){ self.forward(center, c, g); });

}

