#include "libmolgrid/grid_maker.h"
#include "libmolgrid/grid_interpolater.h"
#include "libmolgrid/example_provider.h"
#include "libmolgrid/example.h"
#include "libmolgrid/grid.h"
#include <iostream>
#include <iomanip>
#include <string>

#include <boost/timer/timer.hpp>

using namespace libmolgrid;
using namespace std;


int main(int argc, char *argv[]) {

  if(argc < 2) {
      printf("Need data directory\n");
      exit(-1);
  }

  ExampleProviderSettings settings;
  settings.data_root = string(argv[1]) + "/structs";
  ExampleProvider provider(settings);
  provider.populate(string(argv[1])+"/small.types");
  Example ex;
  provider.next(ex);


  // set up gridmaker and run forward
  GridMaker biggmaker(0.25,41.5);
  float3 grid_dims = biggmaker.get_grid_dims();
  MGrid4f out(ex.num_types(), grid_dims.x, grid_dims.y, grid_dims.z);
  biggmaker.forward(ex, out.gpu());

  GridInterpolater gi(0.25,41.5,0.25,41.5);
  MGrid4f outi(ex.num_types(), grid_dims.x, grid_dims.y, grid_dims.z);

  {
  boost::timer::auto_cpu_timer timer;
  for(unsigned i = 0; i < 100; i++) {
      gi.forward(out.gpu(), outi.gpu(), 2.0, true);
  }
  }
}
