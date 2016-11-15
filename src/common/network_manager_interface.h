/*
The MIT License

Copyright (c) 2015-2016 Albert Murienne

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
#include <vector>

namespace neurocl {

class sample;
class samples_manager;

class network_manager_interface
{
public:

    using t_progress_fct = std::function<void(int)>;

public:

    //! load network topology & weights
    virtual void load_network( const std::string& topology_path, const std::string& weights_path ) = 0;
    //! save network weights
    virtual void save_network() = 0;

    //! train given single sample
    virtual void train( const sample& s ) = 0;
    //! train given samples vector
    virtual void train( const std::vector<sample>& training_set ) = 0;
    //! mini-batch train a samples set
    virtual void batch_train(   const samples_manager& smp_manager,
                                const size_t& epoch_size,
                                const size_t& batch_size,
                                t_progress_fct progress_fct = t_progress_fct() ) = 0;

    // prepare gradient descent
    virtual void prepare_training_iteration() = 0;
    // finalize gradient descent
    virtual void finalize_training_iteration() = 0;
    //! compute network output
    virtual void compute_output( sample& s ) = 0;

    //! dump network parameters
    virtual void dump_weights() = 0;
    virtual void dump_bias() = 0;
    virtual void dump_activations() = 0;
};

} //namespace neurocl

#endif //NETWORK_FACTORY_H
