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
    extractor(settings, defaultGninaReceptorTypes, defaultGninaLigandTypes) {

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


std::shared_ptr<ExampleRefProvider> ExampleProvider::createProvider(const ExampleProviderSettings& settings);



} /* namespace libmolgrid */
