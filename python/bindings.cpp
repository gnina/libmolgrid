/*
 * bindings.cpp
 *
 *  Python bindings for libmolgrid
 */

#include <vector>
#include <type_traits>
#include <utility>
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/tuple.hpp>
#include <boost/preprocessor/repetition.hpp>
#include <boost/preprocessor/punctuation/comma_if.hpp>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "grid.h"
#include "managed_grid.h"
#include "quaternion.h"
#include "transform.h"
#include "atom_typer.h"

using namespace boost::python;
using namespace libmolgrid;

#define TYPEARG(Z, N, T) BOOST_PP_COMMA_IF(N) T
#define NTYPES(N, T) BOOST_PP_REPEAT(N, TYPEARG, T)

/// indicate how MGrid shoudl be automatically converted in python bindings
static bool python_gpu_enabled = true;


//create a grid given the data ptr and dimensions
template<typename GridType, std::size_t ... I>
GridType grid_create(typename GridType::type *data, std::size_t *dims, std::index_sequence<I...>) {
  return GridType(data, dims[I]...);
}

//given a python object, convert to appropriate grid type if possible
//conversions will never copy the underlying grid memory - just retain
//a pointer to it.  The API should not hold onto this pointer since
//its lifetime is managed by python.
template<class Grid_t, bool HasNumpy>
struct Grid_from_python {

    Grid_from_python() {
      //register on construction
      converter::registry::push_back(
          &Grid_from_python::convertible,
          &Grid_from_python::construct,
          type_id<Grid_t>());
    }

    static bool hasattr(object o, const char* name) {
            return PyObject_HasAttrString(o.ptr(), name);
    }

    struct tensor_info {
        //store information extracted from passed type
        void *dataptr;
        size_t shape[LIBMOLGRID_MAX_GRID_DIM];
        size_t ndim;
        bool isdouble;
        bool isGPU;

        tensor_info(): dataptr(nullptr), shape{0,}, ndim(0), isdouble(false), isGPU(false) {}
    };
    //return non-NULL pointer to data and fill out metadata if obj_ptr is torch tensor
    static bool is_torch_tensor(PyObject *obj_ptr, tensor_info& info) {
      handle<> handle(borrowed(obj_ptr));
      object t(handle);
      //basically duck typing
      if(hasattr(t,"data_ptr") && hasattr(t,"shape") && hasattr(t,"type")) {
        long ptrval = extract<long>(t.attr("data_ptr")());
        info.dataptr = (void*)ptrval;
        std::string typ = extract<std::string>(t.attr("type")());
        auto s = tuple(t.attr("shape"));
        info.ndim = len(s);
        for(unsigned i = 0; i < info.ndim; i++) {
          info.shape[i] = extract<size_t>(s[i]);
        }

        if(typ == "torch.FloatTensor") {
          info.isGPU = false;
          info.isdouble = false;
        } else if(typ == "torch.DoubleTensor") {
          info.isGPU = false;
          info.isdouble = true;
        } else if(typ == "torch.cuda.FloatTensor") {
          info.isGPU = true;
          info.isdouble = false;
        } else if(typ == "torch.cuda.DoubleTensor") {
          info.isGPU = true;
          info.isdouble = true;
        } else {
          return false; //don't recognize
        }
        return true;
      }
      return false;
    }

    //return heap allocated tensor_info struct if can convert with all the info
    static void* convertible(PyObject *obj_ptr) {
      tensor_info info;

      extract<typename Grid_t::managed_t> mgrid(obj_ptr);
      if (mgrid.check() && Grid_t::GPU == python_gpu_enabled) {
        Grid_t g = mgrid();
        info.dataptr = g.data();
        info.ndim = Grid_t::N;
        info.isdouble = std::is_same<typename Grid_t::type,double>::value;
        info.isGPU = Grid_t::GPU;

        for(unsigned i = 0; i < info.ndim; i++) {
          info.shape[i] = g.dimension(i);
        }
        return new tensor_info(info);
      }
      else if(is_torch_tensor(obj_ptr, info)) {
        //check correct types
        if(Grid_t::N == info.ndim && Grid_t::GPU == info.isGPU &&
            std::is_same<typename Grid_t::type,double>::value == info.isdouble) {
          return new tensor_info(info);
        }
      } else if(HasNumpy && !Grid_t::GPU && PyArray_Check(obj_ptr)) {
        //numpy array
        auto array = (PyArrayObject*)obj_ptr;
        info.ndim = PyArray_NDIM(array);
        if(Grid_t::N == info.ndim && PyArray_CHKFLAGS(array, NPY_ARRAY_CARRAY)) {
          //check stride? I think CARRAY has stride 1
          //right number of dimensions, check element type
          auto typ = PyArray_TYPE(array);

          info.dataptr = PyArray_DATA(array);
          info.isdouble = (typ == NPY_DOUBLE);
          info.isGPU = false; //numpy always cpu

          auto npdims = PyArray_DIMS(array);
          for(unsigned i = 0; i < info.ndim; i++) {
            info.shape[i] = npdims[i];
          }

          if(typ == NPY_FLOAT && std::is_same<typename Grid_t::type,float>::value) {
            return new tensor_info(info); //should be fine
          } else if(typ == NPY_DOUBLE && std::is_same<typename Grid_t::type,double>::value) {
            return new tensor_info(info);
          }
        }
      }
      return nullptr;
    }

    static void construct(PyObject* obj_ptr,
        boost::python::converter::rvalue_from_python_stage1_data* data) {

      tensor_info info;

      if(data->convertible) { //set to the return vale of convertible
        tensor_info *infop = (tensor_info*)data->convertible;

        //create grid from tensor data
        typedef converter::rvalue_from_python_storage<Grid_t> storage_type;
        void* storage = reinterpret_cast<storage_type*>(data)->storage.bytes;
        data->convertible = new (storage) Grid_t( grid_create<Grid_t>((typename Grid_t::type*)infop->dataptr,
            &infop->shape[0],  std::make_index_sequence<Grid_t::N>()));

        delete infop;
      }
    }
};

//wrapper for float* since it isn't a native python type
template <typename T>
struct Pointer {
    T *ptr;

    Pointer(T *p): ptr(p) {}

    operator T*() const { return ptr; }
};


template<typename GridType,
    typename std::enable_if<
        std::is_same<typename GridType::type, typename GridType::subgrid_t>::value,
        int>::type = 0>
void add_one_dim(class_<GridType>& C) {
  C.def("__setitem__",
      +[](GridType& g, size_t i, typename GridType::type val) {g[i] = val;});
}

template<typename GridType,
    typename std::enable_if<
        !std::is_same<typename GridType::type, typename GridType::subgrid_t>::value,
        int>::type = 0>
void add_one_dim(class_<GridType>& C) {
  //not one-dimensional grid type, do nothing
}

template<typename GridType, std::size_t ... I>
typename GridType::type& grid_get(GridType& g, tuple t,
    std::index_sequence<I...>) {
  return g(static_cast<size_t>(extract<size_t>(t[I]))...);
}

//add common grid methods
template<typename GridType>
void add_grid_members(class_<GridType>& C) {
  C.def(init<GridType>())
      .def("size", &GridType::size)
      .def("dimension", &GridType::dimension)
      .add_property("shape",
          make_function(
          +[](const GridType& g)->tuple {
            return tuple(std::vector<size_t>(g.dimensions(),g.dimensions()+GridType::N)); //hopefully tuple gets move constructed
          }))
      .def("__len__",
          +[](const GridType& g)->size_t {return g.dimension(0);}) //length of first dimension only
      .def("__getitem__",
          +[](const GridType& g, size_t i)-> typename GridType::subgrid_t {return g[i];})
      .def("__getitem__",
          +[](GridType& g, tuple t) -> typename GridType::type {return grid_get(g, t, std::make_index_sequence<GridType::N>());})
      .def("__setitem__",
          +[](GridType& g, tuple t, typename GridType::type val) {grid_get(g, t, std::make_index_sequence<GridType::N>()) = val;})
       // if arguments passed by non-const reference, have to pass grid by value to get bindings to work
      .def("copyTo", +[](const GridType& self, typename GridType::cpu_grid_t dest) { self.copyTo(dest);})
      .def("copyTo", +[](const GridType& self, typename GridType::gpu_grid_t dest) { self.copyTo(dest);})
      .def("copyFrom", static_cast<void (GridType::*)(const typename GridType::cpu_grid_t&)>(&GridType::copyFrom))
      .def("copyFrom", static_cast<void (GridType::*)(const typename GridType::gpu_grid_t&)>(&GridType::copyFrom))
      .def("type", +[](const GridType& g){
              return std::is_same<typename GridType::type,float>::value ? "float32" : std::is_same<typename GridType::type,double>::value  ? "float64" : "unknown";});

}

//register definition for specified grid type
template<class GridType, typename ... Types>
void define_grid(const char* name, bool numpysupport) {

  class_<GridType> C(name, init< Pointer<typename GridType::type>, Types...>());
  add_grid_members(C);
  //setters only for one dimension grids
  add_one_dim(C); //SFINAE!

  if(numpysupport)
    Grid_from_python<GridType, true> convert; //register
  else
    Grid_from_python<GridType, false> convert; //register

}

template<class GridType, typename ... Types>
void define_mgrid(const char* name) {

  class_<GridType> C(name, init<Types...>());
  add_grid_members(C);
  C.def("cpu",static_cast<const typename GridType::cpu_grid_t& (GridType::*)() const>(&GridType::cpu), return_value_policy<copy_const_reference>())
      .def("gpu",static_cast<const typename GridType::gpu_grid_t& (GridType::*)() const>(&GridType::gpu), return_value_policy<copy_const_reference>())
      ;
  //setters only for one dimension grids
  add_one_dim(C); //SFINAE!
}

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(Transform_forward_overloads, Transform::forward, 2, 3)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(Transform_backward_overloads, Transform::backward, 2, 3)

//wrap import array since it includes are return
bool init_numpy()
{
  import_array2("Could not import numpy", false);
  return true;
}

/** \brief Callback to python for index type
 */
class PythonCallbackIndexTyper: public CallbackIndexTyper {

    boost::python::object callback;

  public:

    /// iniitalize callbacktyper, if names are not provided, numerical names will be generated
    PythonCallbackIndexTyper(boost::python::object c, unsigned ntypes,
        const std::vector<std::string>& names=std::vector<std::string>()): CallbackIndexTyper(nullptr, ntypes, names), callback(c) {
    }

    ///call callback
    virtual std::pair<int,float> get_type(OpenBabel::OBAtom& a) const {
      return boost::python::extract< std::pair<int,float> >(callback(a));
    }
};

/** \brief Callback to python for vector type
 * The call back function should return a tuple ([type vector], radius) for the atom
 */
class PythonCallbackVectorTyper: public CallbackVectorTyper {

    boost::python::object callback;

  public:

    /// iniitalize callbacktyper, if names are not provided, numerical names will be generated
    PythonCallbackVectorTyper(boost::python::object c, unsigned ntypes,
        const std::vector<std::string>& names=std::vector<std::string>()): CallbackVectorTyper(nullptr, ntypes, names), callback(c) {
    }

    ///set type vector and return radius for a, not exposed to python
    virtual float get_type(OpenBabel::OBAtom& a, std::vector<float>& typ) const {
      auto vec_r =  get_type_vector(a);
      typ = vec_r.first;
      return vec_r.second;
    }

    ///call callback - for python return vector by reference
    virtual std::pair<std::vector<float>, float > get_type_vector(OpenBabel::OBAtom& a) const {
      return  boost::python::extract< std::pair< std::vector<float>, float > >(callback(a));
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

struct PySwigObject {
    PyObject_HEAD
    void * ptr;
    const char * desc;
};

void* extract_swig_wrapped_pointer(PyObject* obj)
{
  std::cerr << "CAlling extract swig\n";
    //first we need to get the this attribute from the Python Object
    if (!PyObject_HasAttrString(obj, "this"))
        return NULL;
    std::cerr << "has this extract swig\n";

    PyObject* thisAttr = PyObject_GetAttrString(obj, "this");
    if (thisAttr == nullptr)
        return nullptr;
    std::cerr << "got this extract swig\n";

    //This Python Object is a SWIG Wrapper and contains our pointer
    void* pointer = ((PySwigObject*)thisAttr)->ptr;
    Py_DECREF(thisAttr);
    return pointer;
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

// Grids

//Grid bindings
#define DEFINE_GRID_TN(N, TYPE, NAME) define_grid<TYPE,NTYPES(N,unsigned)>(NAME, numpy_supported);
#define DEFINE_GRID(N, CUDA, T) \
DEFINE_GRID_TN(N,Grid##N##T##CUDA, "Grid" #N #T #CUDA)

// MGrid bindings
#define DEFINE_MGRID_TN(N, TYPE, NAME) define_mgrid<TYPE,NTYPES(N,unsigned)>(NAME);
#define DEFINE_MGRID(N, T) \
DEFINE_MGRID_TN(N,MGrid##N##T,"MGrid" #N #T)

//instantiate all dimensions up to and including six
#define DEFINE_GRIDS(Z, N, _) \
DEFINE_GRID(N,CUDA,f) \
DEFINE_GRID(N,CUDA,d) \
DEFINE_GRID(N, ,f) \
DEFINE_GRID(N, ,d) \
DEFINE_MGRID(N,f) \
DEFINE_MGRID(N,d)

  BOOST_PP_REPEAT_FROM_TO(1,LIBMOLGRID_MAX_GRID_DIM, DEFINE_GRIDS, 0);

//vector utility types
  class_<std::vector<size_t> >("SizeVec")
      .def(vector_indexing_suite<std::vector<size_t> >());
  class_<std::vector<std::string> >("StringVec")
      .def(vector_indexing_suite<std::vector<std::string> >());

  class_<Pointer<float> >("FloatPtr", no_init);
  class_<Pointer<double> >("DoublePtr", no_init);

  class_<float3>("float3", no_init)
      .def("__init__",
      make_constructor(
          +[](float x, float y, float z) {return std::make_shared<float3>(make_float3(x,y,z));}))
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
      .def("rotate", &Quaternion::rotate)
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
      .def(init<float3>()) //center
  .def(init<float3, float>()) //center, translate
  .def(init<float3, float, bool>()) //center, translate, rotate
  .def("quaternion", &Transform::quaternion, return_value_policy<copy_const_reference>())
  .def("rotation_center", &Transform::rotation_center)
  .def("translation", &Transform::translation)
  //non-const references need to be passed by value, so wrap
  .def("forward", +[](Transform& self, const Grid2f& in, Grid2f out, bool dotranslate) {self.forward(in,out,dotranslate);},
      Transform_forward_overloads((arg("in"), arg("out"), arg("dotranslate")=true)))
  .def("forward",  +[](Transform& self, const Grid2fCUDA& in, Grid2fCUDA out, bool dotranslate) {self.forward(in,out,dotranslate);},
      Transform_forward_overloads((arg("in"), arg("out"), arg("dotranslate")=true)))
  .def("backward", +[](Transform& self, const Grid2f& in, Grid2f out, bool dotranslate) {self.backward(in,out,dotranslate);},
      Transform_backward_overloads((arg("in"), arg("out"), arg("dotranslate")=true)))
  .def("backward",+[](Transform& self, const Grid2fCUDA& in, Grid2fCUDA out, bool dotranslate) {self.backward(in,out,dotranslate);},
       Transform_backward_overloads((arg("in"), arg("out"), arg("dotranslate")=true)));

//Atom typing
  converter::registry::insert(&extract_swig_wrapped_pointer, type_id<OpenBabel::OBAtom>());
  py_pair<int, float>();

  class_<GninaIndexTyper>("GninaIndexTyper")
      .def(init<bool>())
      .def("num_types", &GninaIndexTyper::num_types)
      .def("get_type", &GninaIndexTyper::get_type)
      .def("get_type_names",&GninaIndexTyper::get_type_names);

  class_<ElementIndexTyper>("ElementIndexTyper")
      .def(init<int>())
      .def("num_types", &ElementIndexTyper::num_types)
      .def("get_type", &ElementIndexTyper::get_type)
      .def("get_type_names",&ElementIndexTyper::get_type_names);

  class_<PythonCallbackIndexTyper>("PythonCallbackIndexTyper", no_init)
      .def(init<object, unsigned>())
      .def(init<object, unsigned, const std::vector<std::string>&>())
      .def("num_types", &PythonCallbackIndexTyper::num_types)
      .def("get_type", &PythonCallbackIndexTyper::get_type)
      .def("get_type_names",&PythonCallbackIndexTyper::get_type_names);

  class_<GninaVectorTyper>("GninaVectorTyper")
      .def("num_types", &GninaVectorTyper::num_types)
      .def("get_type", &GninaVectorTyper::get_type)
      .def("get_type_names",&GninaVectorTyper::get_type_names);

  class_<PythonCallbackVectorTyper>("PythonCallbackVectorTyper", no_init)
      .def(init<object, unsigned>())
      .def(init<object, unsigned, const std::vector<std::string>&>())
      .def("num_types", &PythonCallbackVectorTyper::num_types)
      .def("get_type_vector", &PythonCallbackVectorTyper::get_type_vector)
      .def("get_type_names",&PythonCallbackVectorTyper::get_type_names);

  class_<FileAtomMapper>("FileAtomMapper", init<const std::string&, const std::vector<std::string> >())
      .def("num_types", &FileAtomMapper::num_types)
      .def("get_type", &FileAtomMapper::get_type)
      .def("get_type_names",&FileAtomMapper::get_type_names);

  class_<SubsetAtomMapper>("SubsetAtomMapper", init<const std::vector<int>&, bool>())
      .def(init<const std::vector< std::vector<int> >&, bool>())
      .def("num_types", &SubsetAtomMapper::num_types)
      .def("get_type", &SubsetAtomMapper::get_type)
      .def("get_type_names",&SubsetAtomMapper::get_type_names);

  class_<MappedAtomIndexTyper<SubsetAtomMapper, ElementIndexTyper> >("SubsettedElementTyper",
      init<SubsetAtomMapper, ElementIndexTyper>())
          .def("num_types", &MappedAtomIndexTyper<SubsetAtomMapper, ElementIndexTyper>::num_types)
          .def("get_type", &MappedAtomIndexTyper<SubsetAtomMapper, ElementIndexTyper>::get_type)
          .def("get_type_names",&MappedAtomIndexTyper<SubsetAtomMapper, ElementIndexTyper>::get_type_names);

  class_<MappedAtomIndexTyper<FileAtomMapper, GninaIndexTyper> >("FileMappedGninaTyper",
      init<FileAtomMapper, GninaIndexTyper>())
          .def("num_types", &MappedAtomIndexTyper<FileAtomMapper, GninaIndexTyper> ::num_types)
          .def("get_type", &MappedAtomIndexTyper<FileAtomMapper, GninaIndexTyper> ::get_type)
          .def("get_type_names",&MappedAtomIndexTyper<FileAtomMapper, GninaIndexTyper> ::get_type_names);
}

