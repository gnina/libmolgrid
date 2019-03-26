/*
 * example_provider.cpp
 *
 *  Created on: Mar 26, 2019
 *      Author: dkoes
 */

#include "example_provider.h"

namespace libmolgrid {

ExampleProvider::ExampleProvider(const ExampleProviderSettings& settings):
    provider(createProvider(settings)),
    extractor(settings, defaultGninaReceptorTyper, defaultGninaLigandTyper) {

}


template<typename ...Typers>
ExampleProvider::ExampleProvider(const ExampleProviderSettings& settings, Typers... typs):
  provider(createProvider(settings)), extractor(settings, typs...) {

}

/// use provided provider
ExampleProvider::ExampleProvider(std::shared_ptr<ExampleRefProvider> p, const ExampleExtractor& e): provider(p), extractor(e) {

}


///load example file file fname and setup provider
void ExampleProvider::populate(const std::string& fname, int numLabels, bool hasGroup) {
  ifstream f(fname.c_str());
  if(!f) throw invalid_argument("Could not open file "+fname);
  provider->populate(f, numLabels, hasGroup);
  provider->setup();
}

///load multiple example files
void ExampleProvider::populate(const std::vector<std::string>& fnames, int numLabels, bool hasGroup) {
  for(unsigned i = 0, n = fnames.size(); i < n; i++) {
    ifstream f(fnames[i].c_str());
    if(!f) throw invalid_argument("Could not open file "+fnames[i]);
    provider->populate(f, numLabels, hasGroup);
  }
  provider->setup();
}

///provide next example
void ExampleProvider::next(Example& ex) {
  static ExampleRef ref;
  provider->nextref(ref);
  extractor.extract(ref, ex);
}

///provide a batch of examples
void ExampleProvider::next_batch(std::vector<Example>& ex, unsigned batch_size) {
  static vector<ExampleRef> refs;
  ex.resize(batch_size);
  refs.resize(batch_size);
  for(unsigned i = 0; i < batch_size; i++) {
    provider->nextref(refs[i]);
    extractor.extract(refs[i], ex[i]);
  }

}


std::shared_ptr<ExampleRefProvider> ExampleProvider::createProvider(const ExampleProviderSettings& settings) {
  bool balanced  = settings.balanced;
  bool strat_receptor  = settings.stratify_receptor;
  bool strat_aff = settings.stratify_max != settings.stratify_min;
  bool grouped = settings.max_group_size > 1;

  //strat_aff > strat_receptor > balanced
  if(strat_aff)
  {
    if(strat_receptor)
    {
      if(balanced) // sample 2 from each receptor
      {
        if(grouped)
          return make_shared<GroupedExampleRefProvider<ValueStratifiedExampleRefProfider<ReceptorStratifiedExampleRefProvider<BalancedExampleRefProvider,2> > > >(settings);
        else
          return make_shared<ValueStratifiedExampleRefProfider<ReceptorStratifiedExampleRefProvider<BalancedExampleRefProvider,2> > >(settings);
      }
      else //sample 1 from each receptor
      {
        if(grouped)
          return make_shared<GroupedExampleRefProvider<ValueStratifiedExampleRefProfider<ReceptorStratifiedExampleRefProvider<UniformExampleRefProvider> > > >(settings);
        else
          return make_shared<ValueStratifiedExampleRefProfider<ReceptorStratifiedExampleRefProvider<UniformExampleRefProvider> > >(settings);
      }
    }
    else
    {
      if(balanced)
      {
        if(grouped)
          return make_shared<GroupedExampleRefProvider<ValueStratifiedExampleRefProfider<BalancedExampleRefProvider> > >(settings);
        else
          return make_shared<ValueStratifiedExampleRefProfider<BalancedExampleRefProvider> >(settings);
      }
      else //sample 1 from each receptor
      {
        if(grouped)
          return make_shared<GroupedExampleRefProvider<ValueStratifiedExampleRefProfider<UniformExampleRefProvider> > >(settings);
        else
          return make_shared<ValueStratifiedExampleRefProfider<UniformExampleRefProvider> >(settings);
      }
    }
  }
  else if(strat_receptor)
  {
    if(balanced) // sample 2 from each receptor
    {
      if(grouped)
        return make_shared<GroupedExampleRefProvider<ReceptorStratifiedExampleRefProvider<BalancedExampleRefProvider, 2> > >(settings);
      else
        return make_shared<ReceptorStratifiedExampleRefProvider<BalancedExampleRefProvider, 2> >(settings);
    }
    else //sample 1 from each receptor
    {
      if(balanced)
        return make_shared<GroupedExampleRefProvider<ReceptorStratifiedExampleRefProvider<UniformExampleRefProvider, 1> > >(settings);
      else
        return make_shared<ReceptorStratifiedExampleRefProvider<UniformExampleRefProvider, 1> >(settings);
    }
  }
  else if(balanced)
  {
    if(grouped)
      return make_shared<GroupedExampleRefProvider<BalancedExampleRefProvider> >(settings);
    else
      return make_shared<BalancedExampleRefProvider>(settings);
  }
  else
  {
    if(grouped)
      return make_shared<GroupedExampleRefProvider<UniformExampleRefProvider> >(settings);
    else
      return make_shared<UniformExampleRefProvider>(settings);
  }
}



} /* namespace libmolgrid */
