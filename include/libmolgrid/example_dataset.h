/** \file example_dataset.h
 *  Main class for accessing examples as mappable dataset.
 *
 *
 *  Created on: Apr 12, 2021
 *      Author: dkoes
 */

#ifndef EXAMPLE_DATASET_H_
#define EXAMPLE_DATASET_H_

#include "libmolgrid/example.h"
#include "libmolgrid/exampleref_providers.h"
#include "libmolgrid/example_extractor.h"

namespace libmolgrid {

  // Docstring_ExampleDataset
/** \brief Given a file of examples, provide a "mappable" interface.
 * Although this is configured with ExampleProviderSettings, settings
 * that configure iteration over the dataset are ignored.  It is up to the
 * user to access the data as they see fit.
 * Note that cache_structs is true by default which will load the entirety
 * of the dataset into memory.
 */
class ExampleDataset {
    UniformExampleRefProvider provider;
    mutable ExampleExtractor extractor; //mutable because can cache
    ExampleProviderSettings init_settings; //save settings created with

  public:


    /// Create dataset using default gnina typing
    ExampleDataset(const ExampleProviderSettings& settings=ExampleProviderSettings());

    /// Create dataset/extractor according to settings with single typer
    ExampleDataset(const ExampleProviderSettings& settings, std::shared_ptr<AtomTyper> t);

    /// Create dataset/extractor according to settings with two typers
    ExampleDataset(const ExampleProviderSettings& settings, std::shared_ptr<AtomTyper> t1, std::shared_ptr<AtomTyper> t2);

    /// Create dataset/extractor according to settings
    ExampleDataset(const ExampleProviderSettings& settings, const std::vector<std::shared_ptr<AtomTyper> >& typrs,
        const std::vector<std::string>& molcaches = std::vector<std::string>());

    virtual ~ExampleDataset() {}

    ///load example file file fname and setup provider
    virtual void populate(const std::string& fname, int num_labels=-1);
    ///load multiple example files
    virtual void populate(const std::vector<std::string>& fnames, int num_labels=-1);

    ///return example at position idx
    Example operator[](size_t idx) const;

    ///return number of labels for each example (computed from first example only)
    virtual size_t num_labels() const { return provider.num_labels(); }
    
    ///return settings created with
    const ExampleProviderSettings& settings() const { return init_settings; }

    ///number of types
    size_t num_types() const { return extractor.num_types(); }
    ///names of types (requires explicit typing)
    std::vector<std::string> get_type_names() const { return extractor.get_type_names(); }
    ///return number of examples
    size_t size() const { return provider.size(); }
};

} /* namespace libmolgrid */

#endif /* EXAMPLE_DATASET_H_ */
