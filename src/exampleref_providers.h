/** \file example_providers.h
 *  Various ways to present examples during training.
 *
 *  Created on: Mar 22, 2019
 *      Author: dkoes
 */

#ifndef EXAMPLE_PROVIDERS_H_
#define EXAMPLE_PROVIDERS_H_

#include <iostream>
#include <vector>
#include <algorithm>
#include <type_traits>
#include <unordered_map>
#include <boost/lexical_cast.hpp>
#include "libmolgrid.h"
#include "example.h"

namespace libmolgrid {

using namespace std;


/// abstract class for storing training example references
class ExampleRefProvider {

  public:
    ExampleRefProvider() {}
    ExampleRefProvider(const ExampleProviderSettings& settings) {}
    virtual void addref(const ExampleRef& ex) = 0;
    virtual void setup() = 0; //essentially shuffle if necessary
    virtual void nextref(ExampleRef& ex) = 0;
    virtual unsigned size() const = 0;
    virtual ~ExampleRefProvider() {}

    //read in all the example refs from lines, but does not setup
    virtual int populate(std::istream& lines, int numlabels, bool hasgroup);
};


///single array of examples, possibly shuffled
class UniformExampleRefProvider: public ExampleRefProvider
{
  vector<ExampleRef> all;
  size_t current = 0;
  bool randomize = false;

public:
  UniformExampleRefProvider() {}
  UniformExampleRefProvider(const ExampleProviderSettings& settings): ExampleRefProvider(settings), current(0), randomize(settings.shuffle)
  {
  }

  void addref(const ExampleRef& ex)
  {
    all.push_back(ex);
  }

  void setup()
  {
    current = 0;
    if(randomize) shuffle(all.begin(), all.end(), random_engine);
    if(all.size() == 0) throw invalid_argument("No valid examples found in training set.");
  }

  void nextref(ExampleRef& ex)
  {
    assert(current < all.size());
    ex = all[current];
    current++;
    if(current >= all.size())
    {
      setup(); //reset current and shuffle if necessary
    }
  }

  unsigned size() const { return all.size(); }
};


/// sample uniformly from actives and decoys
class BalancedExampleRefProvider: public ExampleRefProvider
{
  UniformExampleRefProvider actives;
  UniformExampleRefProvider decoys;
  size_t current = 0;
  unsigned labelpos = 0;

public:
  BalancedExampleRefProvider() {}
  BalancedExampleRefProvider(const ExampleProviderSettings& settings):
    ExampleRefProvider(settings), actives(settings), decoys(settings), current(0), labelpos(settings.labelpos)
  {
  }

  void addref(const ExampleRef& ex)
  {
    if(labelpos < ex.labels.size()) {
    if (ex.labels[labelpos])
      actives.addref(ex);
    else
      decoys.addref(ex);
    } else {
      throw invalid_argument("Example has no label at position "+ boost::lexical_cast<string>(labelpos) + ".  There are only "+boost::lexical_cast<string>(ex.labels.size())+" labels");
    }
  }

  void setup()
  {
    current = 0;
    actives.setup();
    decoys.setup();
  }

  void nextref(ExampleRef& ex)
  {
    //alternate between actives and decoys
    if(current % 2 == 0)
      actives.nextref(ex);
    else
      decoys.nextref(ex);

    current++;
  }

  unsigned size() const { return actives.size()+decoys.size(); }

  unsigned num_actives() const { return actives.size(); }
  unsigned num_decoys() const { return decoys.size(); }

  void next_active(ExampleRef& ex)
  {
    actives.nextref(ex);
  }

  void next_decoy(ExampleRef& ex)
  {
    decoys.nextref(ex);
  }
};


/// sample with some specified probability between two providers that should already be initialized
template<class Provider1, class Provider2>
class SamplingExampleRefProvider: public ExampleRefProvider
{
  Provider1 p1;
  Provider2 p2;
  double sample_rate = 0.5;
  std::uniform_real_distribution<double> R{0.0,1};

public:
  SamplingExampleRefProvider() {}
  SamplingExampleRefProvider(const ExampleProviderSettings& settings, Provider1 P1, Provider2 P2, double srate):
    ExampleRefProvider(settings), p1(P1), p2(P2), sample_rate(srate)
  {
  }

  void addref(const ExampleRef& ex)
  {
      throw invalid_argument("Cannot add to SamplingExampleRefProvider");
  }

  void setup()
  {
    p1.setup();
    p2.setup();
  }

  void nextref(ExampleRef& ex)
  {
    //alternate between actives and decoys
    double r = R(random_engine);
    if(r < sample_rate)
      p1.nextref(ex);
    else
      p2.nextref(ex);
  }

  unsigned size() const { return p1.size()+p2.size(); }
};


/** \brief Partition examples by receptor and sample k times uniformly from each receptor
  with k=2 and a balanced_provider you get paired examples from each receptor
  */
template<class Provider, int K=1>
class ReceptorStratifiedExampleRefProvider: public ExampleRefProvider
{
  vector<Provider> examples;
  std::unordered_map<const char*, unsigned> recmap; //map to receptor indices
  ExampleProviderSettings param; //keep copy for instantiating new providers

  size_t currenti = 0; //position in array
  size_t currentk = 0; //number of times sampling it
  bool randomize = false;

public:
  ReceptorStratifiedExampleRefProvider(): currenti(0), currentk(0), randomize(false) {}
  ReceptorStratifiedExampleRefProvider(const ExampleProviderSettings& settings):
    ExampleRefProvider(settings), param(settings), currenti(0), currentk(0), randomize(settings.shuffle)
  {
  }

  void addref(const ExampleRef& ex)
  {
    if(ex.files.size() == 0) {
      throw std::invalid_argument("Missing receptor from line");
    }
    if(recmap.count(ex.files[0]) == 0)
    {
      //allocate
      recmap[ex.files[0]] = examples.size();
      examples.push_back(Provider(param));
    }
    unsigned pos = recmap[ex.files[0]];
    examples[pos].addref(ex);
  }

  //note there is an explicit specialization for balanced providers, k=2
  void setup()
  {
    if(K <= 0) throw std::invalid_argument("Invalid sampling k for ReceptorStratifiedExampleRefProvider");
    currenti = 0; currentk = 0;

    for(unsigned i = 0, n = examples.size(); i < n; i++)
    {
      examples[i].setup();
    }
    //also shuffle receptors
    if(randomize) shuffle(examples.begin(), examples.end(), random_engine);
  }

  void nextref(ExampleRef& ex)
  {
    if(examples.size() == 0) throw std::invalid_argument("No valid stratified examples.");
    if(currentk >= K)
    {
      currentk = 0; //on to next receptor
      currenti++;
    }
    if(currenti >= examples.size())
    {
      currenti = 0;
      if(currentk != 0) std::logic_error("Invalid indices");
      if(randomize) shuffle(examples.begin(), examples.end(), random_engine);
    }

    if(examples[currenti].size() == 0) throw logic_error("No valid sub-stratified examples.");
    examples[currenti].nextref(ex);
    currentk++;
  }

  unsigned size() const
  {
    //no one said this had to be particularly efficient..
    unsigned ret = 0;
    for(unsigned i = 0, n = examples.size(); i < n; i++)
    {
      ret += examples[i].size();
    }
    return ret;
  }
};

template<>
void ReceptorStratifiedExampleRefProvider<BalancedExampleRefProvider, 2>::setup();


/** \brief Partition examples by affinity and sample uniformly from each affinity bin
 affinities are binned by absolute value according to molgriddataparameters
 */
template<class Provider>
class ValueStratifiedExampleRefProfider: public ExampleRefProvider
{
  std::vector<Provider> examples;
  size_t currenti = 0;//position in array
  double min = 0, max = 0, step = 0;
  int valpos = 0;
  bool use_abs = true;

  //return bin for given value
  unsigned bin(double val) const
  {
    if(use_abs)
      val = fabs(val);
    if(val < min) val = min;
    if(val >= max) val = max-FLT_EPSILON;
    val -= min;
    unsigned pos = val/step;
    return pos;
  }
public:
  ValueStratifiedExampleRefProfider() {}
  ValueStratifiedExampleRefProfider(const ExampleProviderSettings& parm):
    ExampleRefProvider(parm), currenti(0)
  {
    max = parm.stratify_max;
    min = parm.stratify_min;
    step = parm.stratify_step;
    use_abs = parm.stratify_abs;
    valpos = parm.stratify_pos;

    if(valpos < 0) throw std::invalid_argument("Negative position for stratification value");
    if(min == max) throw std::invalid_argument("Empty range for value stratification");
    unsigned maxbin = bin(max);
    if(maxbin == 0) throw std::invalid_argument("Not enough bins for value stratification");

    for(unsigned i = 0; i <= maxbin; i++)
    {
      examples.push_back(Provider(parm));
    }
  }

  void addref(const ExampleRef& ex)
  {
    if((unsigned)valpos >= ex.labels.size()) throw std::invalid_argument("Invalid position for value stratification label");
    unsigned i = bin(ex.labels[valpos]);
    if(i >= examples.size()) throw std::invalid_argument("Error with value stratification binning");
    examples[i].addref(ex);
  }

  void setup()
  {
    currenti = 0;
    std::vector<Provider> tmp;
    for(unsigned i = 0, n = examples.size(); i < n; i++)
    {
      if(examples[i].size() > 0) {
        //eliminate empty buckets
        tmp.push_back(std::move(examples[i]));
        tmp.back().setup();
      }
      else {
        LOG(INFO) << "Empty bucket " << i;
      }
    }
    swap(examples,tmp);
    if(examples.size() <= 0)  throw std::invalid_argument("No examples in affinity stratification!");
  }

  void nextref(ExampleRef& ex)
  {
    examples[currenti].nextref(ex);
    currenti = (currenti+1)%examples.size();
  }

  unsigned size() const
  {
    //no one said this had to be particularly efficient..
    unsigned ret = 0;
    for(unsigned i = 0, n = examples.size(); i < n; i++)
    {
      ret += examples[i].size();
    }
    return ret;
  }
};

/** \brief group multiple grids into a single example
 * For example, multiple frames of an MD simulation that will be processed by an RNN.
 * next() returns the next frame for the next example in the batch;
 * traversal is row-major with layout TxN.  Frames are maintained in the order
 * they are added.  There is a fixed batch and time series (group) size.
 * */
template<class Provider>
class GroupedExampleRefProvider : public ExampleRefProvider {

  Provider examples;
  unsigned batch_size = 1;
  unsigned maxgroupsize = 0;

  std::unordered_map<int, std::vector<ExampleRef>> frame_groups;

  unsigned current_ts = 0;
  unsigned current_group_index = 0;
  std::vector<int> current_groups;

public:
  GroupedExampleRefProvider() {     current_groups.assign(batch_size,maxgroupsize); }
  GroupedExampleRefProvider(const ExampleProviderSettings& parm):
                                              ExampleRefProvider(parm), examples(parm),
                                              batch_size(parm.group_batch_size),
                                              maxgroupsize(parm.max_group_size) {
    current_groups.assign(batch_size,maxgroupsize);
  }
  //only add the first example for each group to examples; after that just
  //the filenames to the frame_groups map
  void addref(const ExampleRef& ex) {
    int group = ex.group;

    if(frame_groups.count(group) == 0)  //new group
      examples.addref(ex); //let provider manage, but we are really just using it to select the groups

    frame_groups[group].push_back(ex);
    if(frame_groups[group].size() >= maxgroupsize)
      throw std::invalid_argument("Frame group "+boost::lexical_cast<string>(group)+" is larger than max group size");
  }

  void setup() {
    examples.setup();
  }

  void nextref(ExampleRef& ex) {
    if(current_group_index >= current_groups.size()) {
      current_group_index = 0; //wrap and start next time step
      current_ts++;
    }
    if(current_ts >= maxgroupsize) {
      //wrap around
      current_ts = 0;
    }

    if(current_ts == 0) { //load new groups into batch
      examples.nextref(ex);
      current_groups[current_group_index] = ex.group;
    }

    int group = current_groups[current_group_index];
    auto& timeseries = frame_groups[group];
    //want current_ts from timeseries, but check for truncated series
    if(current_ts < timeseries.size()) {
      ex = timeseries[current_ts];
    } else {
      ex = timeseries.back();
    }

    current_group_index++; //read from next group next
  }

  //return number of groups
  unsigned size() const
  {
    return examples.size();
  }
};

} /* namespace libmolgrid */

#endif /* EXAMPLE_PROVIDERS_H_ */
