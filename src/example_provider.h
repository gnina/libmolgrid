/** \file example_provider.h
 *  Main class for providing examples.
 *
 *
 *  Created on: Mar 26, 2019
 *      Author: dkoes
 */

#ifndef EXAMPLE_PROVIDER_H_
#define EXAMPLE_PROVIDER_H_

#include "example.h"
#include "exampleref_providers.h"
#include "example_extractor.h"

namespace libmolgrid {

/** \brief Given a file of examples, provide Example classes one at a time
 * This contains an exampleref provider, which can be configured using a
 * single settings object if so desired, and an example extractor.
 */
class ExampleProvider {
    std::shared_ptr<ExampleRefProvider> provider;
    ExampleExtractor extractor;

  public:

    /// return provider as specifyed by settings
    static std::shared_ptr<ExampleRefProvider> createProvider(const ExampleProviderSettings& settings);

    /// Create provider using default gnina typing
    ExampleProvider(const ExampleProviderSettings& settings=ExampleProviderSettings());

    /// Create provider/extractor according to settings
    template<typename ...Typers>
    ExampleProvider(const ExampleProviderSettings& settings, Typers... typrs);

    /// use provided provider
    ExampleProvider(std::shared_ptr<ExampleRefProvider> p, const ExampleExtractor& e);
    virtual ~ExampleProvider() {}

    ///load example file file fname and setup provider
    virtual void populate(const std::string& fname, int numLabels=-1, bool hasGroup=false);
    ///load multiple example files
    virtual void populate(const std::vector<std::string>& fnames, int numLabels=-1, bool hasGroup=false);

    ///provide next example
    virtual void next(Example& ex);
    virtual Example next() { Example ex; next(ex); return ex; }

    ///provide a batch of examples
    virtual void next_batch(std::vector<Example>& ex, unsigned batch_size);
    virtual std::vector<Example> next_batch(unsigned batch_size) {
      std::vector<Example> ex;
      next_batch(ex, batch_size);
      return ex;
    }

    ExampleExtractor& get_extractor() { return extractor; }
    ExampleRefProvider& get_provider() { return *provider; }
};

} /* namespace libmolgrid */

#endif /* EXAMPLE_PROVIDER_H_ */
