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

#ifndef NETWORK_MANAGER_MLP_H
#define NETWORK_MANAGER_MLP_H

#include "common/network_factory.h"
#include "common/network_sample.h"

#include <boost/function.hpp>
#include <boost/shared_array.hpp>

#include <vector>

namespace neurocl {

class samples_manager;

namespace mlp {

class network_interface;
class network_file_handler;

class network_manager : public network_manager_interface
{
public:

	typedef boost::function<void(int)> t_progress_fct;

    typedef enum
    {
        MLP_IMPL_BNU_REF = 0,
		MLP_IMPL_BNU_FAST,
        MLP_IMPL_VEXCL
    } t_mlp_impl;

public:

    network_manager( const t_mlp_impl& impl );
	virtual ~network_manager() {}

    void load_network( const std::string& topology_path, const std::string& weights_path );
    void save_network();

    void train( const sample& s );
    void train( const std::vector<sample>& training_set );
    void batch_train(	const samples_manager& smp_manager,
						const size_t& epoch_size,
						const size_t& batch_size,
						t_progress_fct progress_fct = t_progress_fct() );

    // prepare gradient descent
    void prepare_training_iteration();
    // finalize gradient descent
    void finalize_training_iteration();

    void compute_output( sample& s );

    void dump_weights();
    void dump_bias();
    void dump_activations();

private:

    void _assert_loaded();
    void _train( const sample& s );

private:

    bool m_network_loaded;

    boost::shared_ptr<network_interface> m_net;
    boost::shared_ptr<network_file_handler> m_net_file_handler;
};

// used for custom external training
class iterative_trainer
{
public:
    iterative_trainer( network_manager& net_manager, const size_t batch_size )
        : m_net_manager( net_manager ), m_batch_pos( 0 ), m_batch_size( batch_size )
    {
        m_net_manager.prepare_training_iteration();
    }
    virtual ~iterative_trainer()
    {
        m_net_manager.finalize_training_iteration();
        m_net_manager.save_network();
    }

    void train_new( const neurocl::sample& sample )
    {
        m_net_manager.train( sample );

        ++m_batch_pos;

        if ( m_batch_pos >= m_batch_size )
        {
            m_net_manager.finalize_training_iteration();
            m_net_manager.prepare_training_iteration();
            m_batch_pos = 0;
        }
    }

private:
    size_t m_batch_pos;
    size_t m_batch_size;

    network_manager& m_net_manager;
};

} /*namespace neurocl*/ } /*namespace mlp*/

#endif //NETWORK_MANAGER_MLP_H
