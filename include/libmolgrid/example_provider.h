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
 * This contains an ExampleRefProvider, which can be configured using a
 * single settings object if so desired, and an example extractor.
 * Note that cache_structs is true by default which will load the entirety
 * of the dataset into memory.
 */
class ExampleProvider {
    std::shared_ptr<ExampleRefProvider> provider;
    ExampleExtractor extractor;
    ExampleProviderSettings init_settings; //save settings created with
    size_t last_epoch = 0;

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

    /** \brief Return current small epoch number
     *  A small epoch occurs once every training example has been
     *  seen at MOST once.  For example, when providing a balanced
     *  view of unbalanced data, a small epoch will complete once the
     *  less common class has been iterated over.
     *  Note this is the epoch of the next example to be provided, not the previous.
     */
    virtual size_t get_small_epoch_num() const { return provider->get_small_epoch_num(); }

    /** \brief Return current large epoch number
     *  A large epoch occurs once every training example has been
     *  seen at LEAST once.  For example, when providing a balanced
     *  view of unbalanced data, a large epoch will complete once the
     *  more common class has been iterated over.
     *  Note this is the epoch of the next example to be provided, not the previous.
     */
    virtual size_t get_large_epoch_num() const { return provider->get_large_epoch_num(); }

    /// Return number of example in small epoch
    virtual size_t small_epoch_size() const { return provider->small_epoch_size(); }

    /// Return number of example in large epoch
    virtual size_t large_epoch_size() const { return provider->large_epoch_size(); }

    /// Reset to beginning
    virtual void reset() { provider->reset(); }

    ///skip over the first n examples
    virtual void skip(unsigned n);

    ///return settings created with
    const ExampleProviderSettings& settings() const { return init_settings; }


    virtual void next_batch(std::vector<Example>& ex, unsigned batch_size=0);

    ///provide a batch of examples, unspecified or 0 batch_size uses default batch size
    virtual std::vector<Example> next_batch(unsigned batch_size=0) {
      std::vector<Example> ex;
      next_batch(ex, batch_size);
      return ex;
    }

    ///return true if we have crossed into a new epoch (or are about to)
    ///Note that once this returns true once, it won't again until the next
    ///epoch has been consumed.
    bool at_new_epoch() {
      if(init_settings.iteration_scheme == LargeEpoch) {
        size_t e = provider->get_large_epoch_num();
        if(e != last_epoch) {
          last_epoch = e;
          return true;
        }
      } else if(init_settings.iteration_scheme == SmallEpoch) {
        size_t e = provider->get_small_epoch_num();
        if(e != last_epoch) {
          last_epoch = e;
          return true;
        }
      }
      return false;
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
