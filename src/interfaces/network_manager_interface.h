/*
The MIT License

Copyright (c) 2015-2017 Albert Murienne

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef NETWORK_MANAGER_INTERFACE_H
#define NETWORK_MANAGER_INTERFACE_H

#include <functional>
#include <memory>
#include <vector>

namespace neurocl {

class sample;
class samples_manager;
class samples_augmenter;

class network_manager_interface
{
public:

    using t_progress_fct = std::function<void(int)>;

    // keypass idiom to restrict access to set_training & train
    class key_training
    {
    	friend class iterative_trainer;
        key_training() {} key_training( key_training const& ) {}
    };

public:

    //! load network topology & weights
    virtual void load_network( const std::string& topology_path, const std::string& weights_path ) = 0;
    //! save network weights
    virtual void save_network() = 0;

    //! Set training flag
    virtual void set_training( bool training, key_training ) = 0;
    //! train given single sample (training flag IS NOT managed)
    virtual void train( const sample& s, key_training ) = 0;

    //! mini-batch train a samples set (training flag IS managed)
    virtual void batch_train(   const samples_manager& smp_manager,
                                const size_t& epoch_size,
                                const size_t& batch_size,
                                t_progress_fct progress_fct = t_progress_fct() ) = 0;

    // prepare training epoch
    virtual void prepare_training_epoch() = 0;
    // finalize training epoch
    virtual void finalize_training_epoch() = 0;
    //! compute network output for single sample
    virtual void compute_output( sample& s ) = 0;
    //! compute network output using augmented sample
	virtual void compute_augmented_output( sample& s, const std::shared_ptr<samples_augmenter>& smp_augmenter ) = 0;
    //! compute network output for multiple samples
    virtual void compute_output( std::vector<sample>& s ) = 0;

    //! gradient check
	virtual void gradient_check( const sample& s ) = 0;

    //! dump network parameters
    virtual void dump_weights() = 0;
    virtual void dump_bias() = 0;
    virtual void dump_activations() = 0;
};

} //namespace neurocl

#endif //NETWORK_FACTORY_H
