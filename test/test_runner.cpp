#include <boost/program_options.hpp>
#include <iostream>
#include "libmolgrid.h"

#define N_ITERS 10
#define BOOST_TEST_NO_MAIN
#include <boost/test/unit_test.hpp>
#include <cuda_runtime.h>

namespace ut = boost::unit_test;
namespace po = boost::program_options;


void initializeCUDA(int device) {
  cudaError_t error;
  cudaDeviceProp deviceProp;

  error = cudaSetDevice(device);
  if (error != cudaSuccess) {
    std::cerr << "cudaSetDevice returned error code " << error << "\n";
    exit(-1);
  }

  error = cudaGetDevice(&device);

  if (error != cudaSuccess) {
    std::cerr << "cudaGetDevice returned error code " << error << "\n";
    exit(-1);
  }

  error = cudaGetDeviceProperties(&deviceProp, device);

  if (deviceProp.computeMode == cudaComputeModeProhibited) {
    std::cerr
        << "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n";
    exit(-1);
  }

  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties returned error code " << error
        << "\n";
    exit(-1);
  }

}

int main(int argc, char* argv[]) {
  //The following exists so that passing --help prints both the UTF help and our
  //specific program options - I can't see any way of doing this without
  //parsing the args twice, because UTF chomps args it thinks it owns
  unsigned _dumvar1;
  unsigned _dumvar2;
  std::string _dumvar3;
  bool help = false;
  po::positional_options_description positional;
  po::options_description inputs("Input");
  inputs.add_options()
  ("seed", po::value<unsigned>(&_dumvar1),"seed for random number generator")
  ("n_iters", po::value<unsigned>(&_dumvar2),"number of iterations to repeat relevant tests")
  ("log", po::value<std::string>(&_dumvar3),"specify logfile, default is test.log");
  
  po::options_description info("Information");
  info.add_options()("help", po::bool_switch(&help), "print usage information");
  po::options_description desc, desc_simple;
  desc.add(inputs).add(info);
  desc_simple.add(inputs).add(info);
  
  po::variables_map vm;
  try {
    po::store(
        po::command_line_parser(argc, argv).options(desc).style(
            po::command_line_style::default_style
                ^ po::command_line_style::allow_guessing).positional(positional).run(),
        vm);
    notify(vm);
  } catch (po::error& e) {
  }

  if (help) std::cout << desc_simple << '\n';

  initializeCUDA(0);
  
  return 0;
}
