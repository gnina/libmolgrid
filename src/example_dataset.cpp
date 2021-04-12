/*
 * example_dataset.cpp
 *
 *  Created on: Apr 12, 2021
 *      Author: dkoes
 */

#include "libmolgrid/example_dataset.h"
#include "libmolgrid/atom_typer.h"

namespace libmolgrid {

using namespace std;

ExampleDataset::ExampleDataset(const ExampleProviderSettings& settings) :
    provider(settings),
        extractor(settings,
            make_shared < FileMappedGninaTyper > (defaultGninaReceptorTyper),
            make_shared < FileMappedGninaTyper > (defaultGninaLigandTyper)),
        init_settings(settings) {

}

/// Create provider/extractor according to settings with single typer
ExampleDataset::ExampleDataset(const ExampleProviderSettings& settings,
    std::shared_ptr<AtomTyper> t) :
    provider(settings), extractor(settings, t), init_settings(settings) {

}

ExampleDataset::ExampleDataset(const ExampleProviderSettings& settings,
    std::shared_ptr<AtomTyper> t1, std::shared_ptr<AtomTyper> t2) :
    provider(settings), extractor(settings, t1, t2), init_settings(settings) {

}

ExampleDataset::ExampleDataset(const ExampleProviderSettings& settings,
    const std::vector<std::shared_ptr<AtomTyper> >& typrs, const std::vector<std::string>& molcaches)
:
    provider(settings), extractor(settings, typrs, molcaches), init_settings(settings) {

}

///load example file file fname and setup provider
void ExampleDataset::populate(const std::string& fname, int num_labels) {
  ifstream f(fname.c_str());
  if (!f) throw invalid_argument("Could not open file " + fname);
  provider.populate(f, num_labels);
  provider.setup();
}

///load multiple example files
void ExampleDataset::populate(const std::vector<std::string>& fnames, int num_labels) {
  for (unsigned i = 0, n = fnames.size(); i < n; i++) {
    ifstream f(fnames[i].c_str());
    if (!f) throw invalid_argument("Could not open file " + fnames[i]);
    provider.populate(f, num_labels);
  }
  provider.setup();
}

Example ExampleDataset::operator[](size_t idx) const {
    static thread_local ExampleRef ref;
    static thread_local Example ex;
    ref = provider[idx];
    extractor.extract(ref, ex);
    return ex;
}

} /* namespace libmolgrid */
