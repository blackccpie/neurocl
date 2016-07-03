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

#include <boost/range/adaptor/reversed.hpp>

#include <iostream>

namespace neurocl {

lenet::lenet() : m_training_samples( 0 )
{
    float learning_rate = 3.0f/*0.01f*/;
    float weight_decay = 0.f;

    // build optimizer given learning rate and weight decay
    m_optimizer = std::make_shared<optimizer>( learning_rate, weight_decay );
}

void lenet::add_layers()
{
    m_layers.emplace_back( std::make_shared<conv_layer>() );
    m_layers.emplace_back( std::make_shared<full_layer>() );

    // TODO-CNN : harmonize populating prototypes :-(

    /*m_layer_input.populate( 32, 32, 1 );
    m_layer_c1.set_filter_size( 5 ); // 5x5
    m_layer_c1.populate( &m_layer_input, 28, 28, 6 );               //conv
    m_layer_s2.populate( &m_layer_c1, 14, 14, 6 );                  //pool
    m_layer_c3.set_filter_size( 5 ); // 5x5
    m_layer_c3.populate( &m_layer_s2, 10, 10, 16 );                 //conv
    m_layer_s4.populate( &m_layer_c3, 5, 5, 16 );                   //pool
    m_layer_c5.set_filter_size( 5 );
    m_layer_c5.populate( &m_layer_s4, 1, 1, 120 );                  //conv
    m_layer_f6.populate( &m_layer_c5, layer_size( 84, 1 ) );        //full
    m_layer_output.populate( &m_layer_f6, layer_size( 10, 1 ) );

    m_layers = { &m_layer_input, &m_layer_c1, &m_layer_s2, &m_layer_c3, &m_layer_s4, &m_layer_c5, &m_layer_f6, &m_layer_output };*/
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
        std::cout << "--> feed forwarding" << std::endl;
        _layer->feed_forward();
    }
}

void lenet::back_propagate()
{
    for ( auto _layer : boost::adaptors::reverse( m_layers ) )
    {
        _layer->back_propagate();
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
