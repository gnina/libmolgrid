/*
 * example_providers.cpp
 *
 *  Created on: Mar 22, 2019
 *      Author: dkoes
 */

#include "libmolgrid/exampleref_providers.h"
#include <boost/algorithm/string.hpp>

namespace libmolgrid {

using namespace std;
using namespace boost::algorithm;

template<>
void ReceptorStratifiedExampleRefProvider<BalancedExampleRefProvider, 2>::setup() {
  //balanced requires acive and decoys, so filter out receptors that don't have both for convenience
  vector<BalancedExampleRefProvider> tmp;
  currenti = 0; currentk = 0;

  for(unsigned i = 0, n = examples.size(); i < n; i++)
  {
    if(examples[i].num_actives() > 0 && examples[i].num_decoys() > 0) {
      //eliminate empty buckets
      tmp.push_back(examples[i]);
      tmp.back().setup();
    }
    else if(examples[i].num_actives() > 0)
    {
      ExampleRef tmp;
      examples[i].next_active(tmp);
      log(INFO) << "Dropping receptor " << tmp.files[0] << " with no decoys.\n";
    }
    else if(examples[i].num_decoys() > 0)
    {
      ExampleRef tmp;
      examples[i].next_decoy(tmp);
      log(INFO) << "Dropping receptor " << tmp.files[0] << " with no actives.\n";
    }
  }

  swap(examples,tmp);
  if(randomize) shuffle(examples.begin(), examples.end(), random_engine);

}

int ExampleRefProvider::populate(std::istream& lines, int numlabels) {
  if(!lines) throw invalid_argument("Could not read lines");

  string line;
  while (getline(lines, line))
  {
    trim(line);
    if(line.length() > 0) { //ignore blank lines
      ExampleRef ref(line, numlabels, has_group());
      addref(ref);
    }
  }

  return size();
}

void UniformExampleRefProvider::addref(const ExampleRef& ex)
{
  all.push_back(ex);
  nlabels = ex.labels.size();
}

void UniformExampleRefProvider::setup()
{
  current = 0;
  if(randomize) shuffle(all.begin(), all.end(), random_engine);
  if(all.size() == 0) throw std::invalid_argument("No valid examples found in training set.");
}

void UniformExampleRefProvider::nextref(ExampleRef& ex)
{
  assert(current < all.size());
  ex = all[current];

  if(ncopies > 1) {
    current_copy++;
    if(current_copy >= ncopies) {
      current++;
      current_copy = 0;
    }
  } else { //always increment
    current++;
  }
  if(current >= all.size())
  {
    setup(); //reset current and shuffle if necessary
    epoch++;
  }
  cnt++;
}

void BalancedExampleRefProvider::addref(const ExampleRef& ex)
{
  if(labelpos < ex.labels.size()) {
  if (ex.labels[labelpos])
    actives.addref(ex);
  else
    decoys.addref(ex);
  } else {
    throw std::invalid_argument("Example has no label at position "+ itoa(labelpos) + " but a label is required to balance batches.  There are only "+itoa(ex.labels.size())+" labels");
  }
}

void BalancedExampleRefProvider::setup()
{
  current = 0;
  actives.setup();
  decoys.setup();
}

void BalancedExampleRefProvider::nextref(ExampleRef& ex)
{
  //alternate between actives and decoys
  if(current % 2 == 0) {
    actives.nextref(ex);
  } else {
    decoys.nextref(ex);
  }

  current++;
  cnt++;
}


} /* namespace libmolgrid */
