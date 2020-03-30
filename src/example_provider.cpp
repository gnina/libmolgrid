/*
 * example_provider.cpp
 *
 *  Created on: Mar 26, 2019
 *      Author: dkoes
 */

#include "libmolgrid/example_provider.h"
#include "libmolgrid/atom_typer.h"

namespace libmolgrid {

using namespace std;

ExampleProvider::ExampleProvider(const ExampleProviderSettings& settings) :
    provider(createProvider(settings)),
        extractor(settings,
            make_shared < FileMappedGninaTyper > (defaultGninaReceptorTyper),
            make_shared < FileMappedGninaTyper > (defaultGninaLigandTyper)),
        init_settings(settings) {

}

/// Create provider/extractor according to settings with single typer
ExampleProvider::ExampleProvider(const ExampleProviderSettings& settings,
    std::shared_ptr<AtomTyper> t) :
    provider(createProvider(settings)), extractor(settings, t), init_settings(settings) {

}

ExampleProvider::ExampleProvider(const ExampleProviderSettings& settings,
    std::shared_ptr<AtomTyper> t1, std::shared_ptr<AtomTyper> t2) :
    provider(createProvider(settings)), extractor(settings, t1, t2), init_settings(settings) {

}

ExampleProvider::ExampleProvider(const ExampleProviderSettings& settings,
    const std::vector<std::shared_ptr<AtomTyper> >& typrs, const std::vector<std::string>& molcaches)
:
    provider(createProvider(settings)), extractor(settings, typrs, molcaches), init_settings(settings) {

}

/// use provided provider
ExampleProvider::ExampleProvider(std::shared_ptr<ExampleRefProvider> p,
    const ExampleExtractor& e) :
    provider(p), extractor(e) {

}

///load example file file fname and setup provider
void ExampleProvider::populate(const std::string& fname, int num_labels) {
  ifstream f(fname.c_str());
  if (!f) throw invalid_argument("Could not open file " + fname);
  provider->populate(f, num_labels);
  provider->setup();
}

///load multiple example files
void ExampleProvider::populate(const std::vector<std::string>& fnames, int num_labels) {
  for (unsigned i = 0, n = fnames.size(); i < n; i++) {
    ifstream f(fnames[i].c_str());
    if (!f) throw invalid_argument("Could not open file " + fnames[i]);
    provider->populate(f, num_labels);
  }
  provider->setup();
}

///provide next example
void ExampleProvider::next(Example& ex) {
  static thread_local ExampleRef ref;
  provider->nextref(ref);
  extractor.extract(ref, ex);
}

///provide a batch of examples
void ExampleProvider::next_batch(std::vector<Example>& ex, unsigned batch_size) {
  static vector<ExampleRef> refs;
  if(batch_size == 0) {
    batch_size = init_settings.default_batch_size;
  }
  ex.resize(batch_size);
  refs.resize(batch_size);
  provider->check_batch_size(batch_size);
  for (unsigned i = 0; i < batch_size; i++) {
    provider->nextref(refs[i]);
    extractor.extract(refs[i], ex[i]);
  }

}

void ExampleProvider::skip(unsigned n) {
  ExampleRef ref;
  for(unsigned i = 0; i < n; i++) {
    provider->nextref(ref);
  }
}


std::shared_ptr<ExampleRefProvider> ExampleProvider::createProvider(
    const ExampleProviderSettings& settings) {
  bool balanced = settings.balanced;
  bool strat_receptor = settings.stratify_receptor;
  bool strat_aff = settings.stratify_max != settings.stratify_min;
  bool grouped = settings.max_group_size > 1;

  //strat_aff > strat_receptor > balanced
  if (strat_aff)
  {
    if (strat_receptor)
    {
      if (balanced) // sample 2 from each receptor
      {
        if (grouped)
          return make_shared
              < GroupedExampleRefProvider<
                  ValueStratifiedExampleRefProfider<
                      ReceptorStratifiedExampleRefProvider<
                          BalancedExampleRefProvider, 2> > > > (settings);
        else
          return make_shared
              < ValueStratifiedExampleRefProfider<
                  ReceptorStratifiedExampleRefProvider<
                      BalancedExampleRefProvider, 2> > > (settings);
      }
      else //sample 1 from each receptor
      {
        if (grouped)
          return make_shared
              < GroupedExampleRefProvider<
                  ValueStratifiedExampleRefProfider<
                      ReceptorStratifiedExampleRefProvider<
                          UniformExampleRefProvider> > > > (settings);
        else
          return make_shared
              < ValueStratifiedExampleRefProfider<
                  ReceptorStratifiedExampleRefProvider<UniformExampleRefProvider> >
              > (settings);
      }
    }
    else
    {
      if (balanced)
      {
        if (grouped)
          return make_shared
              < GroupedExampleRefProvider<
                  ValueStratifiedExampleRefProfider<BalancedExampleRefProvider> >
              > (settings);
        else
          return make_shared
              < ValueStratifiedExampleRefProfider<BalancedExampleRefProvider>
              > (settings);
      }
      else //sample 1 from each receptor
      {
        if (grouped)
          return make_shared
              < GroupedExampleRefProvider<
                  ValueStratifiedExampleRefProfider<UniformExampleRefProvider> >
              > (settings);
        else
          return make_shared
              < ValueStratifiedExampleRefProfider<UniformExampleRefProvider>
              > (settings);
      }
    }
  }
  else if (strat_receptor)
  {
    if (balanced) // sample 2 from each receptor
    {
      if (grouped)
        return make_shared
            < GroupedExampleRefProvider<
                ReceptorStratifiedExampleRefProvider<BalancedExampleRefProvider,
                    2> > > (settings);
      else
        return make_shared
            < ReceptorStratifiedExampleRefProvider<BalancedExampleRefProvider, 2>
            > (settings);
    }
    else //sample 1 from each receptor
    {
      if (balanced)
        return make_shared
            < GroupedExampleRefProvider<
                ReceptorStratifiedExampleRefProvider<UniformExampleRefProvider,
                    1> > > (settings);
      else
        return make_shared
            < ReceptorStratifiedExampleRefProvider<UniformExampleRefProvider, 1>
            > (settings);
    }
  }
  else if (balanced)
  {
    if (grouped)
      return make_shared < GroupedExampleRefProvider<BalancedExampleRefProvider>
          > (settings);
    else
      return make_shared < BalancedExampleRefProvider > (settings);
  }
  else
  {
    if (grouped)
      return make_shared < GroupedExampleRefProvider<UniformExampleRefProvider>
          > (settings);
    else
      return make_shared < UniformExampleRefProvider > (settings);
  }
}

} /* namespace libmolgrid */
