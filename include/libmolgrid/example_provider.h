/** \file example_provider.h
 *  Main class for providing examples.
 *
 *
 *  Created on: Mar 26, 2019
 *      Author: dkoes
 */

#ifndef EXAMPLE_PROVIDER_H_
#define EXAMPLE_PROVIDER_H_

#include "libmolgrid/example.h"
#include "libmolgrid/exampleref_providers.h"
#include "libmolgrid/example_extractor.h"

namespace libmolgrid {

  // Docstring_ExampleProvider
/** \brief Given a file of examples, provide Example classes one at a time
 * This contains an exampleref provider, which can be configured using a
 * single settings object if so desired, and an example extractor.
 */
class ExampleProvider {
    std::shared_ptr<ExampleRefProvider> provider;
    ExampleExtractor extractor;
    ExampleProviderSettings init_settings; //save settings created with
  public:

    /// return provider as specifyed by settings
    static std::shared_ptr<ExampleRefProvider> createProvider(const ExampleProviderSettings& settings);

    /// Create provider using default gnina typing
    ExampleProvider(const ExampleProviderSettings& settings=ExampleProviderSettings());

    /// Create provider/extractor according to settings with single typer
    ExampleProvider(const ExampleProviderSettings& settings, std::shared_ptr<AtomTyper> t);

    /// Create provider/extractor according to settings with two typers
    ExampleProvider(const ExampleProviderSettings& settings, std::shared_ptr<AtomTyper> t1, std::shared_ptr<AtomTyper> t2);

    /// Create provider/extractor according to settings
    ExampleProvider(const ExampleProviderSettings& settings, const std::vector<std::shared_ptr<AtomTyper> >& typrs,
        const std::vector<std::string>& molcaches = std::vector<std::string>());

    /// use provided provider
    ExampleProvider(std::shared_ptr<ExampleRefProvider> p, const ExampleExtractor& e);
    virtual ~ExampleProvider() {}

    ///load example file file fname and setup provider
    virtual void populate(const std::string& fname, int num_labels=-1);
    ///load multiple example files
    virtual void populate(const std::vector<std::string>& fnames, int num_labels=-1);

    ///return number of labels for each example (computed from first example only)
    virtual size_t num_labels() const { return provider->num_labels(); }
    
    ///provide next example
    virtual void next(Example& ex);
    virtual Example next() { Example ex; next(ex); return ex; }

    ///skip over the first n examples
    virtual void skip(unsigned n);

    ///return settings created with
    const ExampleProviderSettings& settings() const { return init_settings; }

    ///provide a batch of examples, unspecified or 0 batch_size uses default batch size
    virtual void next_batch(std::vector<Example>& ex, unsigned batch_size=0);
    virtual std::vector<Example> next_batch(unsigned batch_size=0) {
      std::vector<Example> ex;
      next_batch(ex, batch_size);
      return ex;
    }

    ExampleExtractor& get_extractor() { return extractor; }
    ExampleRefProvider& get_provider() { return *provider; }

    ///number of types
    size_t num_types() const { return extractor.num_types(); }
    ///names of types (requires explicit typing)
    std::vector<std::string> get_type_names() const { return extractor.get_type_names(); }
    ///return number of examples
    size_t size() const { return provider->size(); }
};

} /* namespace libmolgrid */

#endif /* EXAMPLE_PROVIDER_H_ */
