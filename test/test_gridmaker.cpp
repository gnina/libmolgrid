#define BOOST_TEST_MODULE gridmaker_test
#include <boost/test/unit_test.hpp>
#include "test_util.h"
#include "grid_maker.h"
#include "example_extractor.h"

using namespace libmolgrid;

BOOST_AUTO_TEST_CASE(forward_cpu) {
  //hard-coded example, compared with a reference
  ExampleRef exref("1 none ../data/LIG.mol", 1);
}
