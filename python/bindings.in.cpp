/* @THIS_IS_THE_SOURCE_FILE@
 * bindings.cpp
 *
 *  Python bindings for libmolgrid
 */

#include "bindings.h"
#include "libmolgrid/quaternion.h"
#include "libmolgrid/transform.h"
#include "libmolgrid/atom_typer.h"
#include "libmolgrid/example_provider.h"
#include "libmolgrid/example_dataset.h"
#include "libmolgrid/grid_maker.h"
#include "libmolgrid/grid_interpolater.h"
#include "libmolgrid/grid_io.h"

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
  object ob = import("openbabel.openbabel");
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

//raw constructor for example provider/dataset, args should be typers and kwargs
//settings in exampleprovidersettings
template <typename T>
std::shared_ptr<T> create_ex_provider(tuple args, dict kwargs) {
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
    typers.push_back(extract<std::shared_ptr<AtomTyper> >(args[i]));
  }

  if(N == 0)
    return std::make_shared<T>(settings);
  else
    return std::make_shared<T>(settings, typers);
}


//register a vector of the specified type using name, but only if it isn't already registered
template <typename T>
void register_vector_type(const char *name) {
  type_info info = type_id< std::vector<T> >();
  const converter::registration* reg = converter::registry::query(info);
  if (reg == NULL || (*reg).m_to_python == NULL) {
    //register the type
    class_<std::vector<T> >(name)
        .def(vector_indexing_suite<std::vector<T> >());
  }
}

template <bool isCUDA>
static void vector_sum_types(const std::vector<Example>& self, Grid<float, 2, isCUDA> sum, bool unique_types) {
  if(self.size() != sum.dimension(0)) {
    throw std::invalid_argument("Size of example vector does not match sum grid: "+itoa(self.size())+" vs "+itoa(sum.dimension(0)));
  }
  for(unsigned i = 0, n = self.size(); i < n; i++) {
    Grid<float, 1, isCUDA> subgrid = sum[i];
    self[i].sum_types(subgrid,unique_types);
  }
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
  def("set_gpu_device", +[](int device)->void {LMG_CUDA_CHECK(cudaSetDevice(device));}, "Set current GPU device.");
  def("get_gpu_device", +[]()->int {int device = 0; LMG_CUDA_CHECK(cudaGetDevice(&device)); return device;}, "Get current GPU device.");

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
  register_vector_type<size_t>("SizeVec");
  register_vector_type<std::string>("StringVec");
  register_vector_type<float>("FloatVec");

  class_<std::vector<CoordinateSet> >("CoordinateSetVec")
      .def(vector_indexing_suite<std::vector<CoordinateSet> >());

  class_<std::vector<Example> >("ExampleVec")
      .def("__init__",make_constructor(+[](list l) {
          if(list_is_vec<Example>(l)) {
            return std::make_shared<std::vector<Example> >(list_to_vec<Example>(l));
          } else { //invalid
            throw std::invalid_argument("Need list of examples for ExampleVec");
          }
        }, default_call_policies()))
      .def(vector_indexing_suite<std::vector<Example> >())
      .def("extract_labels", +[](const std::vector<Example>& self, Grid<float, 2, false> out) { Example::extract_labels(self, out);}, "@Docstring_Example_extract_labels@")
      .def("extract_labels", +[](const std::vector<Example>& self, Grid<float, 2, true> out) { Example::extract_labels(self, out);}, "@Docstring_Example_extract_labels@")
      .def("extract_label", +[](const std::vector<Example>& self, int labelpos, Grid<float, 1, false> out) { Example::extract_label(self, labelpos, out);}, "@Docstring_Example_extract_label@")
      .def("extract_label", +[](const std::vector<Example>& self, int labelpos, Grid<float, 1, true> out) { Example::extract_label(self, labelpos, out);}, "@Docstring_Example_extract_label@")
      .def("sum_types", +[](const std::vector<Example>& self, Grid2fCUDA sum, bool unique_types) { vector_sum_types(self, sum, unique_types); }, (arg("sum"), arg("unique_types") = true))
      .def("sum_types", +[](const std::vector<Example>& self, Grid2f sum, bool unique_types) { vector_sum_types(self, sum, unique_types); }, (arg("sum"), arg("unique_types") = true));

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
  class_<Quaternion>("Quaternion", "@Docstring_Quaternion@")
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

  class_<Transform>("Transform", "@Docstring_Transform@")
      .def(init<Quaternion>())
      .def(init<Quaternion, float3>())
      .def(init<Quaternion, float3, float3>())
  .def(init<float3, float, bool>((arg("center"),arg("random_translate")=0.0,arg("random_rotation")=false))) //center, translate, rotate
  .def("get_quaternion", &Transform::get_quaternion, return_value_policy<copy_const_reference>())
  .def("get_rotation_center", &Transform::get_rotation_center)
  .def("get_translation", &Transform::get_translation)
  .def("set_quaternion", &Transform::set_quaternion)
  .def("set_rotation_center", &Transform::set_rotation_center)
  .def("set_translation", &Transform::set_translation)
  //non-const references need to be passed by value, so wrap
  .def("forward", +[](Transform& self, const Grid2f& in, Grid2f out, bool dotranslate) {self.forward(in,out,dotranslate);},
      Transform_forward_overloads("@Docstring_Transform_forward_1@", (arg("in"), arg("out"), arg("dotranslate")=true)))
  .def("forward",  +[](Transform& self, const Grid2fCUDA& in, Grid2fCUDA out, bool dotranslate) {self.forward(in,out,dotranslate);},
      Transform_forward_overloads("@Docstring_Transform_forward_4@", (arg("in"), arg("out"), arg("dotranslate")=true)))
  .def("forward", static_cast<void (Transform::*)(const CoordinateSet&, CoordinateSet&, bool) const>(&Transform::forward),
      Transform_forward_overloads("@Docstring_Transform_forward_3@", (arg("in"), arg("out"), arg("dotranslate")=true)))
  .def("forward", static_cast<void (Transform::*)(const Example&, Example&, bool) const>(&Transform::forward),
          Transform_forward_overloads("@Docstring_Transform_forward_2@", (arg("in"), arg("out"), arg("dotranslate")=true)))
  .def("backward", +[](Transform& self, const Grid2f& in, Grid2f out, bool dotranslate) {self.backward(in,out,dotranslate);},
      Transform_backward_overloads("@Docstring_Transform_backward_1@", (arg("in"), arg("out"), arg("dotranslate")=true)))
  .def("backward",+[](Transform& self, const Grid2fCUDA& in, Grid2fCUDA out, bool dotranslate) {self.backward(in,out,dotranslate);},
       Transform_backward_overloads("@Docstring_Transform_backward_2@", (arg("in"), arg("out"), arg("dotranslate")=true)));

//Atom typing
  converter::registry::insert(&extract_swig_wrapped_pointer, type_id<OpenBabel::OBAtom>());
  converter::registry::insert(&extract_pybel_atom, type_id<OpenBabel::OBAtom>());
  converter::registry::insert(&extract_swig_wrapped_pointer, type_id<OpenBabel::OBMol>());
  converter::registry::insert(&extract_pybel_molecule, type_id<OpenBabel::OBMol>());

  class_<AtomTyper,  std::shared_ptr<AtomTyper> >("AtomTyper", "@Docstring_AtomTyper@", no_init);

  class_<GninaIndexTyper, bases<AtomTyper>, std::shared_ptr<GninaIndexTyper> >("GninaIndexTyper", "@Docstring_GninaIndexTyper@")
      .def(init<bool>())
      .def("num_types", &GninaIndexTyper::num_types)
      .def("get_atom_type_index", &GninaIndexTyper::get_atom_type_index)
      .def("get_type_radii", &GninaIndexTyper::get_type_radii)
      .def("get_type_names",&GninaIndexTyper::get_type_names);
  implicitly_convertible<std::shared_ptr<GninaIndexTyper>, std::shared_ptr<AtomTyper> >();


  class_<ElementIndexTyper, bases<AtomTyper>, std::shared_ptr<ElementIndexTyper> >("ElementIndexTyper", "@Docstring_ElementIndexTyper@")
      .def(init<int>())
      .def("num_types", &ElementIndexTyper::num_types)
      .def("get_atom_type_index", &ElementIndexTyper::get_atom_type_index)
      .def("get_type_radii", &ElementIndexTyper::get_type_radii)
      .def("get_type_names",&ElementIndexTyper::get_type_names);
  implicitly_convertible<std::shared_ptr<ElementIndexTyper>, std::shared_ptr<AtomTyper> >();

  class_<NullIndexTyper, bases<AtomTyper>, std::shared_ptr<NullIndexTyper> >("NullIndexTyper", "@Docstring_NullIndexTyper@")
      .def("num_types", &NullIndexTyper::num_types)
      .def("get_atom_type_index", &NullIndexTyper::get_atom_type_index)
      .def("get_type_radii", &NullIndexTyper::get_type_radii)
      .def("get_type_names",&NullIndexTyper::get_type_names);
  implicitly_convertible<std::shared_ptr<NullIndexTyper>, std::shared_ptr<AtomTyper> >();

  class_<PythonCallbackIndexTyper, bases<AtomTyper>, std::shared_ptr<PythonCallbackIndexTyper> >("PythonCallbackIndexTyper",
      init<object, unsigned, list>(
          (arg("func"), arg("num_types"), arg("names") = list() ) ))
      .def("num_types", &PythonCallbackIndexTyper::num_types)
      .def("get_atom_type_index", &PythonCallbackIndexTyper::get_atom_type_index)
      .def("get_type_names",&PythonCallbackIndexTyper::get_type_names);
  implicitly_convertible<std::shared_ptr<PythonCallbackIndexTyper>, std::shared_ptr<AtomTyper> >();

  class_<GninaVectorTyper, bases<AtomTyper>, std::shared_ptr<GninaVectorTyper> >("GninaVectorTyper", "@Docstring_GninaVectorTyper@")
      .def("num_types", &GninaVectorTyper::num_types)
      .def("get_atom_type_vector", +[](const GninaVectorTyper& typer, OpenBabel::OBAtom* a) {
        std::vector<float> typs;
        float r = typer.get_atom_type_vector(a, typs);
        auto ltyps = list(typs);
        return std::make_pair(ltyps,r);
        })
      .def("get_type_names",&GninaVectorTyper::get_type_names);
  implicitly_convertible<std::shared_ptr<GninaVectorTyper>, std::shared_ptr<AtomTyper> >();

  class_<PythonCallbackVectorTyper, bases<AtomTyper>, std::shared_ptr<PythonCallbackVectorTyper> >("PythonCallbackVectorTyper",
      init<object, unsigned, list>(
          (arg("func"), arg("num_types"), arg("names") = list() ) ))
      .def("num_types", &PythonCallbackVectorTyper::num_types)
      .def("get_atom_type_vector", &PythonCallbackVectorTyper::get_atom_type_vector)
      .def("get_type_names",&PythonCallbackVectorTyper::get_type_names);
  implicitly_convertible<std::shared_ptr<PythonCallbackVectorTyper>, std::shared_ptr<AtomTyper> >();

  class_<FileAtomMapper, std::shared_ptr<FileAtomMapper> >("FileAtomMapper", "@Docstring_FileAtomMapper@", init<const std::string&, const std::vector<std::string> >())
      .def("num_types", &FileAtomMapper::num_types)
      .def("get_new_type", &FileAtomMapper::get_new_type)
      .def("get_type_names",&FileAtomMapper::get_type_names);

  class_<SubsetAtomMapper, std::shared_ptr<SubsetAtomMapper> >("SubsetAtomMapper","@Docstring_SubsetAtomMapper@", init<const std::vector<int>&, bool>())
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

  class_<SubsettedElementTyper, bases<AtomTyper>, std::shared_ptr<SubsettedElementTyper> >("SubsettedElementTyper", "@Docstring_SubsettedElementTyper@", no_init)
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
          .def("get_type_radii",&SubsettedElementTyper::get_type_radii)
          .def("get_type_names",&SubsettedElementTyper::get_type_names);
  implicitly_convertible<std::shared_ptr<SubsettedElementTyper>, std::shared_ptr<AtomTyper> >();

  class_<SubsettedGninaTyper, bases<AtomTyper>, std::shared_ptr<SubsettedGninaTyper> >("SubsettedGninaTyper", no_init)
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
          .def("get_type_radii",&SubsettedGninaTyper::get_type_radii)
          .def("get_type_names",&SubsettedGninaTyper::get_type_names);
  implicitly_convertible<std::shared_ptr<SubsettedGninaTyper>, std::shared_ptr<AtomTyper> >();

  class_<FileMappedGninaTyper, bases<AtomTyper>, std::shared_ptr<FileMappedGninaTyper> >("FileMappedGninaTyper",
          init<const std::string&, bool>((arg("fname"), arg("use_covalent_radius")=false)))
              //todo, add init for file stream inputs if we every want it
          .def("num_types", &FileMappedGninaTyper::num_types)
          .def("get_atom_type_index", &FileMappedGninaTyper::get_atom_type_index)
          .def("get_type_radii",&FileMappedGninaTyper::get_type_radii)
          .def("get_type_names",&FileMappedGninaTyper::get_type_names);
  implicitly_convertible<std::shared_ptr<FileMappedGninaTyper>, std::shared_ptr<AtomTyper> >();

  class_<FileMappedElementTyper, bases<AtomTyper>, std::shared_ptr<FileMappedElementTyper> >("FileMappedElementTyper",
          init<const std::string&, unsigned>((arg("fname"), arg("maxe")=84)))
          .def("num_types", &FileMappedElementTyper::num_types)
          .def("get_atom_type_index", &FileMappedElementTyper::get_atom_type_index)
          .def("get_type_radii",&FileMappedElementTyper::get_type_radii)
          .def("get_type_names",&FileMappedElementTyper::get_type_names);
  implicitly_convertible<std::shared_ptr<FileMappedElementTyper>, std::shared_ptr<AtomTyper> >();

  scope().attr("defaultGninaLigandTyper") = defaultGninaLigandTyper;
  scope().attr("defaultGninaReceptorTyper") = defaultGninaReceptorTyper;


  //molecular data (example providing)
  class_<CoordinateSet>("CoordinateSet", "@Docstring_CoordinateSet@")
      .def(init<OpenBabel::OBMol*, const AtomTyper&>())
      .def(init<OpenBabel::OBMol*>())
      .def(init<const Grid2f&, const Grid1f&, const Grid1f&, unsigned>())
      .def(init<const Grid2fCUDA&, const Grid1fCUDA&, const Grid1fCUDA&, unsigned>())
      .def(init<const Grid2f&, const Grid2f&, const Grid1f&>())
      .def(init<const Grid2fCUDA&, const Grid2fCUDA&, const Grid1fCUDA&>())
      .def(init<const CoordinateSet&, const CoordinateSet&, bool>((arg("rec"), arg("lig"), arg("unique_index_types")=true)))
      .def("has_indexed_types", &CoordinateSet::has_indexed_types)
      .def("has_vector_types", &CoordinateSet::has_vector_types)
      .def("make_vector_types", +[](CoordinateSet& self, bool dummy, list tr) { return self.make_vector_types(dummy,list_to_vec<float>(tr));},
          (arg("include_dummy_type")=false, arg("type_radii") = list()), "@Docstring_CoordinateSet_make_vector_types@")
      .def("make_vector_types", &CoordinateSet::make_vector_types, (arg("include_dummy_type")=false, arg("type_radii") = std::vector<float>()), "@Docstring_CoordinateSet_make_vector_types@")
      .def("size", &CoordinateSet::size)
      .def("num_types", &CoordinateSet::num_types)
      .def("center", &CoordinateSet::center)
      .def("clone", &CoordinateSet::clone)
      .def("togpu", &CoordinateSet::togpu, "set memory affinity to GPU")
      .def("tocpu", &CoordinateSet::tocpu, "set memory affinity to CPU")
      .def("copyTo", +[](const CoordinateSet& self, Grid2f c, Grid1f t, Grid1f r) {return self.copyTo(c,t,r);}, "copy into coord/type/radii grids")
      .def("copyTo", +[](const CoordinateSet& self, Grid2fCUDA c, Grid1fCUDA t, Grid1fCUDA r) {return self.copyTo(c,t,r);}, "copy into coord/type/radii grids")
      .def("copyTo", +[](const CoordinateSet& self, Grid2f c, Grid2f t, Grid1f r) {return self.copyTo(c,t,r);}, "copy into coord/type/radii grids")
      .def("copyTo", +[](const CoordinateSet& self, Grid2fCUDA c, Grid2fCUDA t, Grid1fCUDA r) {return self.copyTo(c,t,r);}, "copy into coord/type/radii grids")
      .def("sum_types", +[](const CoordinateSet& self, Grid1f sum) { self.sum_types(sum);}, "sum types across atoms")
      .def("sum_types", +[](const CoordinateSet& self, Grid1fCUDA sum) { self.sum_types(sum);}, "sum types across atoms")
      .def_readwrite("coords", &CoordinateSet::coords)
      .def_readwrite("type_index", &CoordinateSet::type_index)
      .def_readwrite("type_vector", &CoordinateSet::type_vector)
      .def_readwrite("radii", &CoordinateSet::radii)
      .def_readwrite("max_type", &CoordinateSet::max_type)
      .def_readonly("src", &CoordinateSet::src);


  //mostly exposing this for documentation purposes
#undef EXSET
#define EXSET(TYPE, NAME, DEFAULT, DOC) .def_readwrite(#NAME, &ExampleProviderSettings::NAME, DOC)

  class_<ExampleProviderSettings>("ExampleProviderSettings")
      MAKE_SETTINGS();

  enum_<IterationScheme>("IterationScheme")
      .value("Continuous",Continuous)
      .value("LargeEpoch",LargeEpoch)
      .value("SmallEpoch",SmallEpoch);

  class_<Example>("Example", "@Docstring_Example@")
    .def("num_coordinates", &Example::num_coordinates)
    .def("num_types", &Example::num_types, (arg("unique_index_type")=true))
    .def("merge_coordinates", static_cast<CoordinateSet (Example::*)(unsigned, bool) const>(&Example::merge_coordinates), (arg("start")=0,arg("unique_index_types") = true), "@Docstring_Example_merge_coordinates_1@")
    .def("merge_coordinates", static_cast<void (Example::*)(Grid2f&, Grid1f&, Grid1f&, unsigned, bool) const>(&Example::merge_coordinates), (arg("coord"), "type_index", "radius", arg("start")=0, arg("unique_index_types")=true), "@Docstring_Example_merge_coordinates_2@")
    .def("merge_coordinates", static_cast<void (Example::*)(Grid2f&, Grid2f&, Grid1f&, unsigned, bool) const>(&Example::merge_coordinates), (arg("coord"), "type_vector", "radius", arg("start")=0, arg("unique_index_types")=true), "@Docstring_Example_merge_coordinates_3@")
    .def("togpu", &Example::togpu, "set memory affinity to GPU")
    .def("tocpu", &Example::tocpu, "set memory affinity to CPU")
    .def("has_vector_types", &Example::has_vector_types, (arg("start")=0), "uses vector typing")
    .def("has_index_types", &Example::has_index_types, (arg("start")=0), "uses index typing")
    .def("sum_types", +[](const Example& self, Grid1fCUDA sum, bool unique_types) { self.sum_types(sum,unique_types);}, (arg("sum"), arg("unique_types") = true), "sum types across atoms in coordinate sets")
    .def("sum_types", +[](const Example& self, Grid1f sum, bool unique_types) { self.sum_types(sum,unique_types);}, (arg("sum"), arg("unique_types") = true), "sum types across atoms in coordinate sets")
    .def_readwrite("coord_sets",&Example::sets)
    .def_readwrite("labels",&Example::labels)
    .def_readwrite("group",&Example::group)
    .def_readwrite("seqcont", &Example::seqcont);

  //there is quite a lot of functionality in the C++ api for example providers, but keep it simple in python for now
  class_<ExampleProvider>("ExampleProvider", "@Docstring_ExampleProvider@")
      .def("__init__", raw_constructor(&create_ex_provider<ExampleProvider>,0),"Construct an ExampleProvider using an ExampleSettings object "
          "and the desired AtomTypers for each molecule.  Alternatively, specify individual settings using keyword arguments, where the keys correspond to properties of the ExampleProviderSettings class (please see that class for complete documentation of available settings).")
      .def("populate",
          static_cast<void (ExampleProvider::*)(const std::string&, int)>(&ExampleProvider::populate),
          (arg("file_name"), arg("num_labels")=-1))
      .def("populate", +[](ExampleProvider& self, list l, int num_labels) {
            if(list_is_vec<std::string>(l)) {
                self.populate(list_to_vec<std::string>(l), num_labels);
              } else {
                throw std::invalid_argument("Need list of file names for ExampleProvider");
              }
          },
          (arg("file_names"), arg("num_labels")=-1))
      .def("num_labels", &ExampleProvider::num_labels)
      .def("settings", &ExampleProvider::settings,return_value_policy<copy_const_reference>())
      .def("num_types", &ExampleProvider::num_types)
      .def("size", &ExampleProvider::size)
      .def("get_type_names", &ExampleProvider::get_type_names)
      .def("next", static_cast<Example (ExampleProvider::*)()>(&ExampleProvider::next))
      .def("next_batch", static_cast< std::vector<Example> (ExampleProvider::*)(unsigned)>(&ExampleProvider::next_batch),
          (arg("batch_size")=0))
      .def("get_small_epoch_num", &ExampleProvider::get_small_epoch_num,"Return small epoch number, where an epoch means every example has been seen at MOST once.")
      .def("get_large_epoch_num", &ExampleProvider::get_large_epoch_num, "Return large epoch number, where an epoch means every example has been seen at LEAST once.")
      .def("small_epoch_size", &ExampleProvider::small_epoch_size,"Return size of small epoch")
      .def("large_epoch_size", &ExampleProvider::large_epoch_size,"Return size of large epoch")
      .def("reset", &ExampleProvider::reset, "Reset iterator to beginning")
      .def("__iter__", +[](object self) { return self;})
      .def("__next__", +[](ExampleProvider& self) -> std::vector<Example> {
            if(self.at_new_epoch()) {
              PyErr_SetString(PyExc_StopIteration, "End of epoch.");
              boost::python::throw_error_already_set();
              return std::vector<Example>();
            } else {
              return self.next_batch();
            }
          });


  class_<ExampleDataset>("ExampleDataset", "@Docstring_ExampleDataset@")
      .def("__init__", raw_constructor(&create_ex_provider<ExampleDataset>,0),"Construct an ExampleDataset using an ExampleSettings object "
          "and the desired AtomTypers for each molecule.  Alternatively, specify individual settings using keyword arguments, where the keys correspond to properties of the ExampleProviderSettings class. Settings related to iteration are ignored.")
      .def("populate",
          static_cast<void (ExampleDataset::*)(const std::string&, int)>(&ExampleDataset::populate),
          (arg("file_name"), arg("num_labels")=-1))
      .def("populate", +[](ExampleDataset& self, list l, int num_labels=-1) {
            if(list_is_vec<std::string>(l)) {
                self.populate(list_to_vec<std::string>(l), num_labels);
              } else {
                throw std::invalid_argument("Need list of file names for ExampleProvider");
              }
          },
          (arg("file_names"), arg("num_labels")=-1))
      .def("num_labels", &ExampleDataset::num_labels)
      .def("settings", &ExampleDataset::settings,return_value_policy<copy_const_reference>())
      .def("num_types", &ExampleDataset::num_types)
      .def("size", &ExampleDataset::size)
      .def("__len__",&ExampleDataset::size)
      .def("get_type_names", &ExampleDataset::get_type_names)
      .def("__getitem__", +[](const ExampleDataset& D, int i) {
          if(i < 0) i = D.size()+i; //index from back
          return D[i];
          });

  //grid maker
  class_<GridMaker>("GridMaker", "@Docstring_GridMaker@",
      init<float, float, bool, bool, float, float>(((arg("resolution")=0.5, arg("dimension")=23.5, arg("binary")=false, arg("radius_type_indexed")=false,arg("radius_scale")=1.0), arg("gaussian_radius_multiple")=1.0)))
      .def("spatial_grid_dimensions", +[](GridMaker& self) { float3 dims = self.get_grid_dims(); return make_tuple(int(dims.x),int(dims.y),int(dims.z));})
      .def("grid_dimensions", +[](GridMaker& self, int ntypes) { float3 dims = self.get_grid_dims(); return make_tuple(ntypes,int(dims.x),int(dims.y),int(dims.z));})
      .def("get_resolution", &GridMaker::get_resolution)
      .def("set_resolution", &GridMaker::set_resolution)
      .def("get_dimension", &GridMaker::get_dimension)
      .def("set_dimension", &GridMaker::set_dimension)
      .def("get_binary", &GridMaker::get_binary)
      .def("set_binary", &GridMaker::set_binary)
      .def("get_radii_type_indexed", &GridMaker::get_radii_type_indexed)
      .def("set_radii_type_indexed", &GridMaker::set_radii_type_indexed)
      //grids need to be passed by value
      .def("forward", +[](GridMaker& self, const Example& ex, Grid<float, 4, false> g, float random_translate, bool random_rotate){
            self.forward(ex, g, random_translate, random_rotate); },
            (arg("example"),arg("grid"),arg("random_translation")=0.0,arg("random_rotation")=false), "\n Generate CPU grid tensor from an example.\n Coordinates may be optionally translated/rotated.  Do not use this function\n if it is desirable to retain the transformation used (e.g., when backpropagating).\n The center of the last coordinate set before transformation\n will be used as the grid center.\n\n:param in:  example\n:param out:  a 4D grid\n:param random_translation:   maximum amount to randomly translate each coordinate (+/-)\n:param random_rotation:  whether or not to randomly rotate\n")           
      .def("forward", +[](GridMaker& self, const Example& ex, Grid<float, 4, true> g, float random_translate, bool random_rotate){
            self.forward(ex, g, random_translate, random_rotate); },
            (arg("example"),arg("grid"),arg("random_translation")=0.0,arg("random_rotation")=false), "\n Generate GPU grid tensor from an example.\n Coordinates may be optionally translated/rotated.  Do not use this function\n if it is desirable to retain the transformation used (e.g., when backpropagating).\n The center of the last coordinate set before transformation\n will be used as the grid center.\n\n:param in:  example\n:param out:  a 4D grid\n:param random_translation:   maximum amount to randomly translate each coordinate (+/-)\n:param random_rotation:  whether or not to randomly rotate\n")           
      .def("forward", +[](GridMaker& self, const std::vector<Example>& in, Grid<float, 5, false> g, float random_translate, bool random_rotate){
            self.forward(in, g, random_translate, random_rotate); },
            (arg("examplevec"),arg("grid"),arg("random_translation")=0.0,arg("random_rotation")=false), "@Docstring_GridMaker_forward_5@")
      .def("forward", +[](GridMaker& self, const std::vector<Example>& in, Grid<float, 5, true> g, float random_translate, bool random_rotate){
            self.forward(in, g, random_translate, random_rotate); },
            (arg("examples"),arg("grid"),arg("random_translation")=0.0,arg("random_rotation")=false), "@Docstring_GridMaker_forward_5@")
      .def("forward", +[](GridMaker& self, float3 center, const CoordinateSet& c, Grid<float, 4, false> g){ self.forward(center, c, g); }, "@Docstring_GridMaker_forward_1@")
      .def("forward", +[](GridMaker& self, float3 center, const CoordinateSet& c, Grid<float, 4, true> g){ self.forward(center, c, g); }, "@Docstring_GridMaker_forward_2@")
      .def("forward", +[](GridMaker& self, const Example& ex, const Transform& t, Grid<float, 4, false> g){ self.forward(ex, t, g); }, "@Docstring_GridMaker_forward_3@")
      .def("forward", +[](GridMaker& self, const Example& ex, const Transform& t, Grid<float, 4, true> g){ self.forward(ex, t, g); }, "@Docstring_GridMaker_forward_3@")
      .def("forward", +[](GridMaker& self, float3 grid_center, const Grid<float, 2, false>& coords,
        const Grid<float, 1, false>& type_index, const Grid<float, 1, false>& radii,
        Grid<float, 4, false>& out){ self.forward(grid_center, coords, type_index, radii, out);}, "@Docstring_GridMaker_forward_6@")
      .def("forward", +[](GridMaker& self, float3 grid_center, const Grid<float, 2, true>& coords,
          const Grid<float, 1, true>& type_index, const Grid<float, 1, true>& radii,
          Grid<float, 4, true>& out){ self.forward(grid_center, coords, type_index, radii, out);}, "@Docstring_GridMaker_forward_7@")
      .def("forward", +[](GridMaker& self, float3 grid_center, const Grid<float, 2, false>& coords,
          const Grid<float, 2, false>& type_vector, const Grid<float, 1, false>& radii,
          Grid<float, 4, false> g){ self.forward(grid_center, coords, type_vector, radii, g); }, "@Docstring_GridMaker_forward_8@")
      .def("forward", +[](GridMaker& self, float3 grid_center, const Grid<float, 2, true>& coords,
              const Grid<float, 2, true>& type_vector, const Grid<float, 1, true>& radii,
              Grid<float, 4, true> g){ self.forward(grid_center, coords, type_vector, radii, g); }, "@Docstring_GridMaker_forward_9@")
      .def("forward", +[](GridMaker& self, const Grid<float, 2, false>& centers, const Grid<float, 3, false>& coords,
              const Grid<float, 2, false>& types, const Grid<float, 2, false>& radii,
              Grid<float, 5, false> g){ self.forward<float, 2, false>(centers, coords, types, radii, g); }, "@Docstring_GridMaker_forward_10@")
      .def("forward", +[](GridMaker& self, const Grid<float, 2, true>& centers, const Grid<float, 3, true>& coords,
              const Grid<float, 2, true>& types, const Grid<float, 2, true>& radii,
              Grid<float, 5, true> g){ self.forward<float, 2, true>(centers, coords, types, radii, g); }, "@Docstring_GridMaker_forward_10@")
      .def("forward", +[](GridMaker& self, const Grid<float, 2, false>& centers, const Grid<float, 3, false>& coords,
              const Grid<float, 3, false>& types, const Grid<float, 2, false>& radii,
              Grid<float, 5, false> g){ self.forward<float, 3, false>(centers, coords, types, radii, g); }, "@Docstring_GridMaker_forward_10@")
      .def("forward", +[](GridMaker& self, const Grid<float, 2, true>& centers, const Grid<float, 3, true>& coords,
              const Grid<float, 3, true>& types, const Grid<float, 2, true>& radii,
              Grid<float, 5, true> g){ self.forward<float, 3, true>(centers, coords, types, radii, g); }, "@Docstring_GridMaker_forward_10@")
      .def("backward", +[](GridMaker& self, float3 grid_center, const CoordinateSet& in, const Grid<float, 4, false>& diff,
          Grid<float, 2, false> atomic_gradients, Grid<float, 2, false> type_gradients){
          self.backward(grid_center, in, diff, atomic_gradients, type_gradients);}, "@Docstring_GridMaker_backward_1@")
      .def("backward", +[](GridMaker& self, float3 grid_center, const CoordinateSet& in,
          const Grid<float, 4, false>& diff, Grid<float, 2, false> atomic_gradients) {
          self.backward(grid_center, in, diff, atomic_gradients); }, "@Docstring_GridMaker_backward_2@")
      .def("backward", +[](GridMaker& self, float3 grid_center, const CoordinateSet& in, const Grid<float, 4, true>& diff,
          Grid<float, 2, true> atomic_gradients, Grid<float, 2, true> type_gradients){
          self.backward(grid_center, in, diff, atomic_gradients, type_gradients);}, "@Docstring_GridMaker_backward_3@")
      .def("backward", +[](GridMaker& self, float3 grid_center, const CoordinateSet& in,
          const Grid<float, 4, true>& diff, Grid<float, 2, true> atomic_gradients) {
          self.backward(grid_center, in, diff, atomic_gradients); }, "@Docstring_GridMaker_backward_4@")
       .def("backward", +[](GridMaker& self, float3 grid_center, const Grid<float, 2, false>& coords,
           const Grid<float, 1, false>& type_index, const Grid<float, 1, false>& radii,
           const Grid<float, 4, false>& diff, Grid<float, 2, false> atom_gradients) {
           self.backward(grid_center, coords, type_index, radii, diff, atom_gradients);}, "@Docstring_GridMaker_backward_5@")
       .def("backward", +[](GridMaker& self, float3 grid_center, const Grid<float, 2, true>& coords,
           const Grid<float, 1, true>& type_index, const Grid<float, 1, true>& radii,
           const Grid<float, 4, true>& diff, Grid<float, 2, true> atom_gradients) {
           self.backward(grid_center, coords, type_index, radii, diff, atom_gradients);}, "@Docstring_GridMaker_backward_6@")
       .def("backward", +[](GridMaker& self, float3 grid_center, const Grid<float, 2, false>& coords,
           const Grid<float, 2, false>& type_vectors, const Grid<float, 1, false>& radii,
           const Grid<float, 4, false>& diff, Grid<float, 2, false> atom_gradients, Grid<float, 2, false> type_gradients) {
              self.backward(grid_center, coords, type_vectors, radii, diff, atom_gradients, type_gradients);}, "@Docstring_GridMaker_backward_7@")
       .def("backward", +[](GridMaker& self, float3 grid_center, const Grid<float, 2, true>& coords,
           const Grid<float, 2, true>& type_vectors, const Grid<float, 1, true>& radii,
           const Grid<float, 4, true>& diff, Grid<float, 2, true> atom_gradients, Grid<float, 2, true> type_gradients) {
              self.backward(grid_center, coords, type_vectors, radii, diff, atom_gradients, type_gradients);}, "@Docstring_GridMaker_backward_8@")
       .def("backward_gradients", +[](GridMaker& self, float3 grid_center,  const Grid<float, 2, false>& coords,
                      const Grid<float, 2, false>& type_vectors, const Grid<float, 1, false>& radii, const Grid<float, 4, false>& diff,
                      const Grid<float, 2, false>& atom_gradients, const Grid<float, 2, false>& type_gradients,
                      Grid<float, 4, false> diffdiff, Grid<float, 2, false> atom_diffdiff, Grid<float, 2, false> type_diffdiff) {
              self.backward_gradients(grid_center, coords, type_vectors, radii, diff, atom_gradients, type_gradients,
                  diffdiff, atom_diffdiff, type_diffdiff); }, "@Docstring_GridMaker_backward_gradients_1@")
       .def("backward_gradients", +[](GridMaker& self, float3 grid_center,  const Grid<float, 2, true>& coords,
                       const Grid<float, 2, true>& type_vectors, const Grid<float, 1, true>& radii, const Grid<float, 4, true>& diff,
                       const Grid<float, 2, true>& atom_gradients, const Grid<float, 2, true>& type_gradients,
                       Grid<float, 4, true> diffdiff, Grid<float, 2, true> atom_diffdiff, Grid<float, 2, true> type_diffdiff) {
               self.backward_gradients(grid_center, coords, type_vectors, radii, diff, atom_gradients, type_gradients,
                   diffdiff, atom_diffdiff, type_diffdiff); }, "@Docstring_GridMaker_backward_gradients_2@")
       .def("backward_gradients", +[](GridMaker& self, float3 grid_center,  const CoordinateSet& in, const Grid<float, 4, false>& diff,
                      const Grid<float, 2, false>& atom_gradients, const Grid<float, 2, false>& type_gradients,
                      Grid<float, 4, false> diffdiff, Grid<float, 2, false> atom_diffdiff, Grid<float, 2, false> type_diffdiff) {
              self.backward_gradients(grid_center, in, diff, atom_gradients, type_gradients,
                  diffdiff, atom_diffdiff, type_diffdiff); }, "@Docstring_GridMaker_backward_gradients_3@")
       .def("backward_gradients", +[](GridMaker& self, float3 grid_center,  const CoordinateSet& in, const Grid<float, 4, true>& diff,
                       const Grid<float, 2, true>& atom_gradients, const Grid<float, 2, true>& type_gradients,
                       Grid<float, 4, true> diffdiff, Grid<float, 2, true> atom_diffdiff, Grid<float, 2, true> type_diffdiff) {
               self.backward_gradients(grid_center, in, diff, atom_gradients, type_gradients,
                   diffdiff, atom_diffdiff, type_diffdiff); }, "@Docstring_GridMaker_backward_gradients_4@");

  class_<GridInterpolater>("GridInterpolater", "@Docstring_GridInterpolater@",
      init<float, float, float, float>((arg("in_resolution")=0.5, arg("in_dimension")=23.5,  arg("out_resolution")=0.5, arg("out_dimension")=23.5)))
      .def("get_in_resolution", &GridInterpolater::get_in_resolution)
      .def("set_in_resolution", &GridInterpolater::set_in_resolution)
      .def("get_out_resolution", &GridInterpolater::get_out_resolution)
      .def("set_out_resolution", &GridInterpolater::set_out_resolution)
      .def("get_in_dimension", &GridInterpolater::get_in_dimension)
      .def("set_in_dimension", &GridInterpolater::set_in_dimension)
      .def("get_out_dimension", &GridInterpolater::get_out_dimension)
      .def("set_out_dimension", &GridInterpolater::set_out_dimension)
      .def("forward", +[](GridInterpolater& self, const Grid<float, 4, false>& in, const Transform& transform, Grid<float, 4, false>& out){
            self.forward(in, transform, out);}, "@Docstring_GridInterpolater_forward_1@")
      .def("forward", +[](GridInterpolater& self, const Grid<float, 4, true>& in, const Transform& transform, Grid<float, 4, true>& out){
            self.forward(in, transform, out);}, "@Docstring_GridInterpolater_forward_2@")
      .def("forward", +[](GridInterpolater& self, const Grid<float, 4, false>& in, const Transform& transform, Grid<float, 4, false>& out){
            self.forward(in, transform, out);}, "@Docstring_GridInterpolater_forward_3@")
      .def("forward", +[](GridInterpolater& self, const Grid<float, 4, true>& in, const Transform& transform, Grid<float, 4, true>& out){
            self.forward(in, transform, out);}, "@Docstring_GridInterpolater_forward_4@")
      .def("forward", +[](GridInterpolater& self, float3 in_center, const Grid<float, 4, false>& in, const Transform& transform,
                                                  float3 out_center, Grid<float, 4, false>& out){
            self.forward(in_center, in, transform, out_center, out);}, "@Docstring_GridInterpolater_forward_5@")
      .def("forward", +[](GridInterpolater& self, float3 in_center, const Grid<float, 4, true>& in, const Transform& transform,
                                                  float3 out_center, Grid<float, 4, true>& out){
            self.forward(in_center, in, transform, out_center, out);}, "@Docstring_GridInterpolater_forward_6@");


  class_<CartesianGrid<MGrid3f> >("CartesianGrid", "@Docstring_CartesianGrid@", init<MGrid3f, float3, float>())
      .def("center",&CartesianGrid<MGrid3f>::center)
      .def("resolution", &CartesianGrid<MGrid3f>::resolution)
      .def("grid", +[](CartesianGrid<MGrid3f>& self) { return self.grid();});

  //grid io
  def("read_dx",static_cast<CartesianGrid<ManagedGrid<float, 3> > (*)(const std::string&)>(&read_dx<float>), "@Docstring_read_dx@");
  def("write_dx",static_cast<void (*)(const std::string& fname, const Grid3f&, const float3&, float, float)>(&write_dx<float>),
      (arg("file_name"),"grid","center","resolution",arg("scale")=1.0), "@Docstring_write_dx@");
  def("write_map",static_cast<void (*)(const std::string& fname, const Grid3f&, const float3&, float, float)>(&write_map<float>),
      (arg("file_name"),"grid","center","resolution",arg("scale")=1.0), "@Docstring_write_map@");
  def("write_dx_grids",static_cast<void (*)(const std::string&, const std::vector<std::string>&, const Grid4f&, const float3&, float, float)>(&write_dx_grids<float>),
      (arg("prefix"),"type_names","grid","center","resolution",arg("scale")=1.0), "@Docstring_write_dx_grids@");
  def("read_dx_grids",+[](const std::string& prefix, const std::vector<std::string>& names, Grid4f grid) { read_dx_grids(prefix, names, grid);}, "@Docstring_read_dx_grids@");


}
