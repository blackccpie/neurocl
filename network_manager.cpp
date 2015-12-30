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

#include "network.h"
#include "network_bnu.h"
#include "network_manager.h"
#include "network_exception.h"

#include <boost/foreach.hpp>
#include <boost/make_shared.hpp>

namespace neurocl {

network_manager::network_manager( const t_neural_impl& impl ) : m_network_loaded( false )
{
    switch( impl )
    {
    case NEURAL_IMPL_BNU:
        m_net = boost::make_shared<network_bnu>();
        break;
    case NEURAL_IMPL_VEXCL:
        m_net = boost::make_shared<network>();
        break;
    }
}

void network_manager::load_network( const std::string& name )
{
    std::vector<size_t> layer_sizes;
    layer_sizes.push_back( 16 ); // input L0
    layer_sizes.push_back( 12 ); // L1
    layer_sizes.push_back( 8 ); // L2
    layer_sizes.push_back( 4 ); // L3
    layer_sizes.push_back( 2 ); // L4
    layer_sizes.push_back( 1 ); // output L5
    m_net->add_layers_2d( layer_sizes );

    m_network_loaded = true;

    std::cout << "network_manager::load_network - network loaded" << std::endl;
}

void network_manager::save_network()
{
    if ( !m_network_loaded )
        throw network_exception( "no network loaded!" );
}

void network_manager::train( const std::vector<sample>& training_set )
{
    if ( !m_network_loaded )
        throw network_exception( "no network loaded!" );

    size_t index = 0;

    BOOST_FOREACH( const neurocl::sample& s, training_set )
    {
        std::cout << "network_manager::train - training sample " << (index+1) << "/" << training_set.size() << std::endl;

        m_net->set_input_sample( s.isample_size, s.isample, s.osample_size, s.osample );

        m_net->feed_forward();
        std::cout << "output before gd = " << m_net->output() << std::endl;
        m_net->gradient_descent();
        m_net->feed_forward();
        std::cout << "output after gd = " << m_net->output() << std::endl;

        std::cout << "network_manager::train - feed_forward & gradient descent successfull for training sample " << index << std::endl;

        ++index;

        std::cout << "network_manager::train - training successfull" << std::endl;
    }
}

void network_manager::compute_output( const sample& s )
{
    m_net->set_input_sample( s.isample_size, s.isample, s.osample_size, s.osample );
    m_net->feed_forward();
}

}; //namespace neurocl