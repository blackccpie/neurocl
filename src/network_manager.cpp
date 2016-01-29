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
#include "network_bnu.h"
#include "network_manager.h"
#include "network_exception.h"
#include "network_file_handler.h"

#include <boost/chrono.hpp>
#include <boost/foreach.hpp>
#include <boost/make_shared.hpp>

namespace neurocl {

void network_manager::_assert_loaded()
{
    if ( !m_network_loaded )
        throw network_exception( "no network loaded!" );
}

network_manager::network_manager( const t_neural_impl& impl ) : m_network_loaded( false )
{
    switch( impl )
    {
    case NEURAL_IMPL_BNU:
        m_net = boost::make_shared<network_bnu>();
        break;
    case NEURAL_IMPL_VEXCL:
        m_net = boost::make_shared<network_vexcl>();
        break;
    }

    m_net_file_handler = boost::make_shared<network_file_handler>( m_net );
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
    _assert_loaded();

    m_net->prepare_training();
}

void network_manager::finalize_training_iteration()
{
    _assert_loaded();

    m_net->update_params();
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
        std::cout << "network_manager::train - training sample " << (index+1) << "/" << training_set.size() << std::endl;

        _train( s );

        ++index;
    }
}

void network_manager::_train( const sample& s )
{
    namespace bc = boost::chrono;
    bc::system_clock::time_point start = bc::system_clock::now();
    bc::milliseconds duration;

    m_net->set_input( s.isample_size, s.isample );
    m_net->set_output( s.osample_size, s.osample );
    duration = bc::duration_cast<bc::milliseconds>( bc::system_clock::now() - start );
    std::cout << "sample set at " << duration.count() << "ms"<< std::endl;

    m_net->feed_forward();
    duration = bc::duration_cast<bc::milliseconds>( bc::system_clock::now() - start );
    std::cout << "ff before gd at " << duration.count() << "ms"<< std::endl;

    m_net->back_propagate();
    duration = bc::duration_cast<bc::milliseconds>( bc::system_clock::now() - start );
    std::cout << "bp at " << duration.count() << "ms"<< std::endl;

    duration = boost::chrono::duration_cast<bc::milliseconds>( bc::system_clock::now() - start );
    std::cout << "network_manager::_train - training successfull in "  << duration.count() << "ms"<< std::endl;
}

void network_manager::compute_output( sample& s )
{
    _assert_loaded();

    m_net->set_input( s.isample_size, s.isample );
    m_net->feed_forward();
    output_ptr output_layer = m_net->output();
    // TODO-AM : not usefull, pass the sample to net?
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

}; //namespace neurocl
