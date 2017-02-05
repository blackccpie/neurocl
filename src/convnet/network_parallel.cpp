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

#include "network_parallel.h"
#include "layer.h"
#include "tensor_tank.h"
#include "tensor_solver.h"

#include "common/thread_pool.h"

namespace neurocl { namespace convnet {

network_parallel::network_parallel( const size_t tasks_size )
    : m_tasks_size( tasks_size ), m_current_net( 0 )
{
    if ( m_tasks_size > MAX_PARRALLEL_TASKS )
        throw network_exception( "maximum parallel tasks number is limited to 10 for now" );
    else
        LOGGER(info) << "network_parallel::network_parallel - " << m_tasks_size <<
            " concurrent threads will be managed" << std::endl;

    layer::set_shared( true );

    m_thread_pool.reset( new thread_pool{ m_tasks_size } );

    m_solver = tensor_solver_factory::build();

    m_networks.reserve( m_tasks_size );

    for ( size_t i = 0; i < m_tasks_size; i++ )
    {
        m_networks.emplace_back();
        m_parallel_jobs.emplace_back( std::bind( &network_parallel::_feed_back, this, i ) );
    }
}

network_parallel::~network_parallel()
{
    layer::set_shared( false );
}

void network_parallel::set_training( bool training )
{
    layer::set_training( training );
}

void network_parallel::add_layers( const std::vector<layer_descr>& layers )
{
    size_t i = 0;
    for ( auto& _network : m_networks )
    {
        LOGGER(info) << "network_parallel::add_layers - adding layers for net #" << i++ << "(" << &_network << ")" << std::endl;
        _network.add_layers( layers );
    }
}

void network_parallel::set_input(  const size_t& in_size, const float* in )
{
    if ( layer::get_training() )
    {
    	m_networks[m_current_net].set_input( in_size, in );
    }
    else
        return m_networks.at(0).set_input( in_size, in );
}

void network_parallel::set_output( const size_t& out_size, const float* out )
{
    m_networks[m_current_net].set_output( out_size, out );
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
    {
        m_thread_pool->add_job( m_parallel_jobs.at( m_current_net++ ) );
        m_current_net = ( m_current_net == m_tasks_size ) ? 0 : m_current_net;
    }
    else
        _feed_back( 0 );
}

void network_parallel::back_propagate()
{
    // TODO-CNN : not very clear but backprop is included in feed_forwarding task dispatching
}

void network_parallel::_feed_back( const size_t i )
{
    std::lock_guard<std::mutex> guard( m_mutex[i] );

    m_networks[i].feed_forward();
    m_networks[i].back_propagate();
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

float network_parallel::loss()
{
    return m_networks.at(0).loss();
}

void network_parallel::gradient_check( const output_ptr& out_ref )
{
    LOGGER(error) << "network_parallel::gradient_check - not implemented for parallel training" << std::endl;
}

} /*namespace neurocl*/ } /*namespace convnet*/
