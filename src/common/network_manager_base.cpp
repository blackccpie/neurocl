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

#include "network_manager_base.h"
#include "network_base_interface.h"
#include "network_file_handler_interface.h"

#include "common/network_exception.h"
#include "common/samples_manager.h"
#include "common/logger.h"

#include <chrono>
#include <iostream>

//#define TRAIN_CHRONO

namespace neurocl {

void network_manager_base::_assert_loaded()
{
    if ( !m_network_loaded )
        throw network_exception( "no network loaded!" );
}

void network_manager_base::load_network( const std::string& topology_path, const std::string& weights_path )
{
    m_net_file_handler->load_network_topology( topology_path );
    m_net_file_handler->load_network_weights( weights_path );

    m_network_loaded = true;

    LOGGER(info) << "network_manager_base::load_network - network loaded" << std::endl;
}

void network_manager_base::save_network()
{
    _assert_loaded();

    m_net_file_handler->save_network_weights();
}

void network_manager_base::batch_train( const samples_manager& smp_manager,
                                        const size_t& epoch_size,
                                        const size_t& batch_size,
                                        t_progress_fct progress_fct )
{
    _assert_loaded();

    size_t progress_size = 0;
    const size_t pbm_size = epoch_size * smp_manager.samples_size();

    for ( size_t i=0; i<epoch_size; i++ )
    {
        LOGGER(info) << std::endl << "network_manager_base::batch_train - EPOCH " << (i+1) << "/" << epoch_size << std::endl;

        while ( true )
        {
            std::vector<neurocl::sample> samples = smp_manager.get_next_batch( batch_size );

            // end of training set management
            if ( samples.empty() )
                break;

            prepare_training_iteration();
            train( samples );
            finalize_training_iteration();

            progress_size += samples.size();

			int progress = ( ( 100 * progress_size ) / pbm_size );

			if ( progress_fct )
				progress_fct( progress );

            std::cout << "\network_manager_base::batch_train - progress " << progress << "%";// << std::endl;
        }

        smp_manager.rewind();
        smp_manager.shuffle();
    }

    std::cout << std::endl;

    save_network();
}

void network_manager_base::prepare_training_iteration()
{
    m_net->clear_gradients();
}

void network_manager_base::finalize_training_iteration()
{
    m_net->gradient_descent();
}

void network_manager_base::train( const sample& s )
{
    _assert_loaded();

    _train( s );
}

void network_manager_base::train( const std::vector<sample>& training_set )
{
    _assert_loaded();

    size_t index = 0;

    for( const auto& s : training_set )
    {
        //LOGGER(info) << "network_manager_base::train - training sample " << (index+1) << "/" << training_set.size() << std::endl;

        _train( s );

        ++index;
    }
}

void network_manager_base::_train( const sample& s )
{
#ifdef TRAIN_CHRONO
    namespace sc = std::chrono;
    sc::system_clock::time_point start = sc::system_clock::now();
    sc::milliseconds duration;
#endif

    // set input/output
    m_net->set_input( s.isample_size, s.isample );
    m_net->set_output( s.osample_size, s.osample );

    // forward/backward propagation
    m_net->feed_forward();
    m_net->back_propagate();

#ifdef TRAIN_CHRONO
    duration = sc::duration_cast<sc::milliseconds>( sc::system_clock::now() - start );
    LOGGER(info) << "network_manager_base::_train - training successfull in "  << duration.count() << "ms"<< std::endl;
#endif
}

void network_manager_base::compute_output( sample& s )
{
    _assert_loaded();

    m_net->set_input( s.isample_size, s.isample );
    m_net->feed_forward();
    output_ptr output_layer = m_net->output();
    std::copy( output_layer.outputs.get(), output_layer.outputs.get() + output_layer.num_outputs, const_cast<float*>( s.osample ) );
}

void network_manager_base::dump_weights()
{
    _assert_loaded();

    std::cout << m_net->dump_weights();
}

void network_manager_base::dump_bias()
{
    _assert_loaded();

    std::cout << m_net->dump_bias();
}

void network_manager_base::dump_activations()
{
    _assert_loaded();

    std::cout << m_net->dump_activations();
}

} /*namespace neurocl*/
