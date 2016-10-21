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

#include "network_vexcl.h"
#include "network_bnu_ref.h"
#include "network_manager.h"
#include "network_file_handler.h"

#ifdef SIMD_ENABLED
    #include "network_bnu_fast.h"
#endif

#include "common/network_exception.h"
#include "common/samples_manager.h"

#include <boost/chrono.hpp>
#include <boost/foreach.hpp>
#include <boost/make_shared.hpp>

//#define TRAIN_CHRONO

namespace neurocl { namespace mlp {

void network_manager::_assert_loaded()
{
    if ( !m_network_loaded )
        throw network_exception( "no network loaded!" );
}

network_manager::network_manager( const t_mlp_impl& impl ) : m_network_loaded( false )
{
    switch( impl )
    {
    case t_mlp_impl::MLP_IMPL_BNU_REF:
        m_net = std::make_shared<network_bnu_ref>();
        break;
    case t_mlp_impl::MLP_IMPL_BNU_FAST:
#ifdef SIMD_ENABLED
        m_net = std::make_shared<network_bnu_fast>();
#else
        throw network_exception( "unmanaged mlp implementation (simd disabled)!" );
#endif
        break;
    case t_mlp_impl::MLP_IMPL_VEXCL:
        m_net = std::make_shared<network_vexcl>();
        break;
    default:
        throw network_exception( "unmanaged mlp implementation!" );
    }

    m_net_file_handler = std::make_shared<network_file_handler>( m_net );
}

void network_manager::load_network( const std::string& topology_path, const std::string& weights_path )
{
    m_net_file_handler->load_network_topology( topology_path );
    m_net_file_handler->load_network_weights( weights_path );

    m_network_loaded = true;

    std::cout << "network_manager::load_network - network loaded" << std::endl;
}

void network_manager::save_network()
{
    _assert_loaded();

    m_net_file_handler->save_network_weights();
}

void network_manager::prepare_training_iteration()
{
    m_net->prepare_training();
}

void network_manager::finalize_training_iteration()
{
    m_net->gradient_descent();
}

void network_manager::train( const sample& s )
{
    _assert_loaded();

    _train( s );
}

void network_manager::train( const std::vector<sample>& training_set )
{
    _assert_loaded();

    size_t index = 0;

    BOOST_FOREACH( const neurocl::sample& s, training_set )
    {
        //std::cout << "network_manager::train - training sample " << (index+1) << "/" << training_set.size() << std::endl;

        _train( s );

        ++index;
    }
}

void network_manager::batch_train( 	const samples_manager& smp_manager,
									const size_t& epoch_size,
									const size_t& batch_size,
									t_progress_fct progress_fct )
{
    _assert_loaded();

    size_t progress_size = 0;
    const size_t pbm_size = epoch_size * smp_manager.samples_size();

    for ( size_t i=0; i<epoch_size; i++ )
    {
        std::cout << std::endl << "network_manager::batch_train - EPOCH " << (i+1) << "/" << epoch_size << std::endl;

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

            std::cout << "\rnetwork_manager::batch_train - progress " << progress << "%";// << std::endl;
        }

        smp_manager.rewind();
    }

    std::cout << std::endl;

    save_network();
}

void network_manager::_train( const sample& s )
{
#ifdef TRAIN_CHRONO
    namespace bc = boost::chrono;
    bc::system_clock::time_point start = bc::system_clock::now();
    bc::milliseconds duration;
#endif

    // set input/output
    m_net->set_input( s.isample_size, s.isample );
    m_net->set_output( s.osample_size, s.osample );

    // forward/backward propagation
    m_net->feed_forward();
    m_net->back_propagate();

#ifdef TRAIN_CHRONO
    duration = boost::chrono::duration_cast<bc::milliseconds>( bc::system_clock::now() - start );
    std::cout << "network_manager::_train - training successfull in "  << duration.count() << "ms"<< std::endl;
#endif
}

void network_manager::compute_output( sample& s )
{
    _assert_loaded();

    m_net->set_input( s.isample_size, s.isample );
    m_net->feed_forward();
    output_ptr output_layer = m_net->output();
    std::copy( output_layer.outputs.get(), output_layer.outputs.get() + output_layer.num_outputs, const_cast<float*>( s.osample ) );
}

void network_manager::dump_weights()
{
    _assert_loaded();

    std::cout << m_net->dump_weights();
}

void network_manager::dump_bias()
{
    _assert_loaded();

    std::cout << m_net->dump_bias();
}

void network_manager::dump_activations()
{
    _assert_loaded();

    std::cout << m_net->dump_activations();
}

} /*namespace neurocl*/ } /*namespace mlp*/
