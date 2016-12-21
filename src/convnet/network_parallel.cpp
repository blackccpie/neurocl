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

#include "network_parallel.h"
#include "layer.h"
#include "tensor_tank.h"
#include "tensor_solver.h"

#include "common/thread_pool.h"

namespace neurocl { namespace convnet {

#define parallel_thread_count 5

network_parallel::network_parallel()
    : m_thread_pool( new thread_pool{ parallel_thread_count } ), m_current_net( 0 )
{
    layer::set_shared( true );

    m_solver = tensor_solver_factory::build();

    for ( size_t i = 0; i < parallel_thread_count; i++ )
        m_networks.emplace_back( network{} );
}

network_parallel::~network_parallel()
{
    layer::set_shared( false );
}

void network_parallel::set_training( bool training )
{
    m_networks.at(0).set_training( training );
}

void network_parallel::add_layers( const std::vector<layer_descr>& layers )
{
    size_t i = 0;
    for ( auto& _network : m_networks )
    {
        LOGGER(info) << "network_parallel::add_layers - adding layers for net #" << i++ << std::endl;
        _network.add_layers( layers );
    }
}

void network_parallel::set_input(  const size_t& in_size, const float* in )
{
    if ( layer::get_training() )
    {
    	for ( auto& _network : m_networks )
    	{
        	_network.set_input( in_size, in );
    	}
    }
    else
        return m_networks.at(0).set_input( in_size, in );
}

void network_parallel::set_output( const size_t& out_size, const float* out )
{
    for ( auto& _network : m_networks )
    {
        _network.set_output( out_size, out );
    }
}

const size_t network_parallel::count_layers()
{
    return m_networks.at(0).count_layers();
}

const layer_ptr network_parallel::get_layer_ptr( const size_t layer_idx )
{
    return m_networks.at(0).get_layer_ptr( layer_idx );
}

void network_parallel::set_layer_ptr( const size_t layer_idx, const layer_ptr& l )
{
    for ( auto& _network : m_networks )
    {
    	_network.set_layer_ptr( layer_idx, l );
    }
}

const output_ptr network_parallel::output()
{
    return std::move( m_networks.at(0).output() );
}

void network_parallel::clear_gradients()
{
    for ( auto& _network : m_networks )
    {
        _network.clear_gradients();
    }
}

void network_parallel::feed_forward()
{
    if ( layer::get_training() )
        m_thread_pool->add_job( std::bind( &network_parallel::_feed_back, this, &m_networks.at(m_current_net++) ) );
    else
        _feed_back( &m_networks.at(0) );
}

void network_parallel::back_propagate()
{
    // TODO-CNN : not very clear but backprop is included in feed_forwarding task dispatching
}

void network_parallel::_feed_back( network* net )
{
    net->feed_forward();
    net->back_propagate();
}

void network_parallel::gradient_descent()
{
    m_thread_pool->wait_all();

    // gradient accumulation
    tensor_tank::instance().accumulate();

    // gradient descent
    m_networks.at(0).gradient_descent();

	// reset net index
    m_current_net = 0;
}

void network_parallel::gradient_check( const output_ptr& out_ref )
{
    LOGGER(error) << "network_parallel::gradient_check - not implemented for parallel training" << std::endl;
}

} /*namespace neurocl*/ } /*namespace convnet*/
