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

#include "lenet.h"
#include "optimizer.h"
#include "conv_layer.h"
#include "full_layer.h"
#include "pool_layer.h"
#include "input_layer.h"
#include "output_layer.h"

#include <boost/range/adaptor/reversed.hpp>

#include <iostream>

namespace neurocl {

//#define VERBOSE_LENET

lenet::lenet() : m_training_samples( 0 )
{
    float learning_rate = 3.0f/*0.01f*/;
    float weight_decay = 0.f;

    // build optimizer given learning rate and weight decay
    m_optimizer = std::make_shared<optimizer>( learning_rate, weight_decay );
}

void lenet::add_layers()
{
    std::shared_ptr<input_layer> in = std::make_shared<input_layer>();
    in->populate( 32, 32, 1 );
    m_layers.emplace_back( in );

    std::shared_ptr<conv_layer> c1 = std::make_shared<conv_layer>( "c1" );
    c1->set_filter_size( 5 ); // 5x5
    c1->populate( m_layers.back(), 28, 28, 6 );
    m_layers.emplace_back( c1 );

    std::shared_ptr<pool_layer> s2 = std::make_shared<pool_layer>( "s2" );
    s2->populate( m_layers.back(), 14, 14, 6 );
    m_layers.emplace_back( s2 );

    std::shared_ptr<conv_layer> c3 = std::make_shared<conv_layer>( "c3" );
    c3->set_filter_size( 5 ); // 5x5
    c3->populate( m_layers.back(), 10, 10, 16 );
    m_layers.emplace_back( c3 );

    std::shared_ptr<pool_layer> s4 = std::make_shared<pool_layer>( "s4" );
    s4->populate( m_layers.back(), 5, 5, 16 );
    m_layers.emplace_back( s4 );

    std::shared_ptr<conv_layer> c5 = std::make_shared<conv_layer>( "c5" );
    c5->set_filter_size( 5 ); // 5x5
    c5->populate( m_layers.back(), 1, 1, 120 );
    m_layers.emplace_back( c5 );

    std::shared_ptr<full_layer> f6 = std::make_shared<full_layer>( "f6" );
    f6->populate( m_layers.back(), 84, 1, 1 );
    m_layers.emplace_back( f6 );

    std::shared_ptr<output_layer> out = std::make_shared<output_layer>();
    out->populate( m_layers.back(), 10, 1, 1 );
    m_layers.emplace_back( out );
}

void lenet::set_input(  const size_t& in_size, const float* in )
{
    // TODO-CNN : for now works only because input layer has no depth for now

    std::shared_ptr<layer> layer;
    std::shared_ptr<input_layer> input_layer = std::dynamic_pointer_cast<neurocl::input_layer>( m_layers.front() );

    if ( in_size > input_layer->size() )
        throw network_exception( "sample size exceeds allocated layer size!" );

#ifdef VERBOSE_LENET
    std::cout << "lenet::set_input - input (" << in << ") size = " << in_size << std::endl;
#endif

    input_layer->fill( 0, 0, in_size, in );
}

void lenet::set_output( const size_t& out_size, const float* out )
{
    // TODO-CNN : for now works only because output layer has no depth for now

    std::shared_ptr<output_layer> output_layer = std::dynamic_pointer_cast<neurocl::output_layer>( m_layers.back() );

    if ( out_size > output_layer->size() )
        throw network_exception( "output size exceeds allocated layer size!" );

#ifdef VERBOSE_LENET
    std::cout << "lenet::set_output - output (" << out << ") size = " << out_size << std::endl;
#endif

    output_layer->fill( 0, 0, out_size, out );
}

const output_ptr lenet::output()
{
    // TODO-CNN : for now works only because last layer has no depth for now

    std::shared_ptr<output_layer> output_layer = std::dynamic_pointer_cast<neurocl::output_layer>( m_layers.back() );

    output_ptr o( output_layer->width() * output_layer->height() );
    output_layer->fill( 0, 0, o.outputs.get() );

    return o;
}

void lenet::prepare_training()
{
    for ( auto _layer : m_layers )
    {
        _layer->prepare_training();
    }

    m_training_samples = 0;
}

void lenet::feed_forward()
{
    for ( auto _layer : m_layers )
    {
#ifdef VERBOSE_LENET
        std::cout << "--> feed forwarding " << _layer->type() << " layer" << std::endl;
#endif
        _layer->feed_forward();
    }
}

void lenet::back_propagate()
{
    for ( auto _layer : boost::adaptors::reverse( m_layers ) )
    {
#ifdef VERBOSE_LENET
        std::cout << "--> back propagating " << _layer->type() << " layer" << std::endl;
#endif
        _layer->back_propagate();
    }

    for ( auto _layer : m_layers )
    {
#ifdef VERBOSE_LENET
        std::cout << "--> updating gradients " << _layer->type() << " layer" << std::endl;
#endif
        _layer->update_gradients();
    }

    ++m_training_samples;
}

void lenet::gradient_descent()
{
    m_optimizer->set_size( m_training_samples );

    for ( auto _layer : m_layers )
    {
        _layer->gradient_descent( m_optimizer );
    }
}

}; //namespace neurocl
