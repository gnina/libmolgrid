/*
 * bindings_gridss.cpp
 *
 * In an effort to speed compilation, put grid instantiation here.
 *  Created on: Mar 27, 2019
 *      Author: dkoes
 */

#include "bindings.h"

/* Holy guacamole batman - the new "improved" numpy api seems to rely on
 * static inline functions serving as unique identifiers, which obviously
 * they won't be if you have separate compilation units.  So major issues
 * (mysterious segfaults) if you include arrayobject in different files.
 * Basically, any numpy trickier has to happen in this file.
 */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

using namespace boost::python;
using namespace libmolgrid;

//wrap import array since it includes are return
bool init_numpy()
{
  import_array2("Could not import numpy", false);
  return true;
}

//create a grid given the data ptr and dimensions
template<typename GridType, std::size_t ... I>
GridType grid_create(typename GridType::type *data, std::size_t *dims, std::index_sequence<I...>) {
  return GridType(data, dims[I]...);
}

template<typename GridType,
    typename std::enable_if<
        std::is_same<typename GridType::type, typename GridType::subgrid_t>::value,
        int>::type = 0>
void add_one_dim(boost::python::class_<GridType>& C) {
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
        if(info.isGPU && hasattr(t,"device") && hasattr(t.attr("device"), "index")) {
          int d = extract<int>(t.attr("device").attr("index"));
          int currd = 0;
          LMG_CUDA_CHECK(cudaGetDevice(&currd));
          if(currd != d) {
            throw std::invalid_argument("Attempt to use GPU tensor on different device ("+itoa(d)+") than current device ("+itoa(currd)+").  Change location of tensor or change current device.");
          }
        }
        if(hasattr(t,"is_contiguous")) {
          if(!t.attr("is_contiguous")()) {
            throw std::invalid_argument("Attempt to use non-contiguous tensor in molgrid.  Call clone first.");
            return false;
          }
        }
        return true;
      }
      return false;
    }

    //return heap allocated tensor_info struct if can convert with all the info
    static void* convertible(PyObject *obj_ptr) {
      tensor_info info;

      if(obj_ptr == nullptr) return nullptr;

      //convert from managed to grid
      extract<typename Grid_t::managed_t> mgrid(obj_ptr);
      if (mgrid.check() && Grid_t::GPU == python_gpu_enabled) {
        typename Grid_t::managed_t mg = mgrid();
        if(Grid_t::GPU)
          info.dataptr = mg.gpu().data();
        else
          info.dataptr = mg.cpu().data();
        info.ndim = Grid_t::N;
        info.isdouble = std::is_same<typename Grid_t::type,double>::value;
        info.isGPU = Grid_t::GPU;

        for(unsigned i = 0; i < info.ndim; i++) {
          info.shape[i] = mg.dimension(i);
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

//add common grid methods
template<typename GridType>
void add_grid_members(class_<GridType>& C) {
  C.def(init<GridType>())
      .def("size", &GridType::size)
      .def("dimension", &GridType::dimension)
      .def("data", +[](const GridType& self) { return (size_t)self.data();}) //more for debugging
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
      .def("copyTo", +[](const GridType& self, typename GridType::cpu_grid_t dest) { return self.copyTo(dest);})
      .def("copyTo", +[](const GridType& self, typename GridType::gpu_grid_t dest) { return self.copyTo(dest);})
      .def("copyFrom", static_cast<size_t (GridType::*)(const typename GridType::cpu_grid_t&)>(&GridType::copyFrom))
      .def("copyFrom", static_cast<size_t (GridType::*)(const typename GridType::gpu_grid_t&)>(&GridType::copyFrom))
      .def("fill_zero", &GridType::fill_zero)
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
      .def("clone", &GridType::clone)
      .def("ongpu", &GridType::ongpu)
      .def("oncpu", &GridType::oncpu)
      .def("copyTo", +[](const GridType& self, GridType dest) {return self.copyTo(dest);})
      .def("copyFrom", static_cast<size_t (GridType::*)(const typename GridType::base_t&)>(&GridType::copyFrom))
      ;
  //setters only for one dimension grids
  add_one_dim(C); //SFINAE!
}


//Grid bindings
#undef MAKE_GRID_TN
#define MAKE_GRID_TN(N, TYPE, NAME) template void define_grid<TYPE,NTYPES(N,unsigned)>(const char*, bool);

// MGrid bindings
#undef MAKE_MGRID_TN
#define MAKE_MGRID_TN(N, TYPE, NAME) template void define_mgrid<TYPE,NTYPES(N,unsigned)>(const char *);

MAKE_ALL_GRIDS();

