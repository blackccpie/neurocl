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

#include "lenet_bnu.h"
#include "network_config.h"
#include "network_exception.h"
#include "network_utils.h"

#include <boost/optional.hpp>
#include <boost/make_shared.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/range/adaptor/reversed.hpp>

namespace bnu = boost::numeric::ublas;

namespace neurocl {

lenet_bnu::lenet_bnu() : m_training_samples( 0 )
{
    float learning_rate = 3.0f/*0.01f*/;
    float weight_decay = 0.f;

    const network_config& nc = network_config::instance();
    nc.update_optional( "learning_rate", learning_rate );
    nc.update_optional( "weight_decay", weight_decay );

    // build optimizer given learning rate and weight decay
    m_optimizer = boost::make_shared<optimizer>( learning_rate, weight_decay );
}

void lenet_bnu::set_input(  const size_t& in_size, const float* in )
{
    if ( in_size > m_layers[0]->size() )
        throw network_exception( "sample size exceeds allocated layer size!" );

    std::cout << "lenet_bnu::set_input - input (" << in << ") size = " << in_size << std::endl;

    // TODO-CNN : for now works only because first layer is one dimensional

    auto input_feature_map = m_layers[0]->feature_map(0).data();
    std::copy( in, in + in_size, input_feature_map.begin() );
}

void lenet_bnu::set_output( const size_t& out_size, const float* out )
{
    // TODO-CNN : for now works only because last layer is one dimensional fcnn

    if ( out_size > m_training_output.size() )
        throw network_exception( "output size exceeds allocated layer size!" );

    std::cout << "lenet_bnu::set_output - output (" << out << ") size = " << out_size << std::endl;

    std::copy( out, out + out_size, m_training_output.begin() );
}

void lenet_bnu::add_layers_2d( const std::vector<layer_size>& layer_sizes )
{
    // TODO-CNN : harmonize populating prototypes :-(

    m_layer_input.populate( 32, 32, 1 );
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

    m_layers = { &m_layer_input, &m_layer_c1, &m_layer_s2, &m_layer_c3, &m_layer_s4, &m_layer_c5, &m_layer_f6, &m_layer_output };
}

const layer_ptr lenet_bnu::get_layer_ptr( const size_t layer_idx )
{
    // TODO-CNN
    /*if ( layer_idx >= m_layers.size() )
    {
        std::cerr << "network_bnu_base::get_layer_ptr - cannot access layer " << layer_idx << std::endl;
        throw network_exception( "invalid layer index" );
    }

    matrixF& weights = m_layers[layer_idx].weights();
    vectorF& bias = m_layers[layer_idx].bias();
    layer_ptr l( weights.size1() * weights.size2(), bias.size() );
    std::copy( &weights.data()[0], &weights.data()[0] + ( weights.size1() * weights.size2() ), l.weights.get() );
    std::copy( &bias[0], &bias[0] + bias.size(), l.bias.get() );

    return l;*/
    return layer_ptr(0,0);
}

void lenet_bnu::set_layer_ptr( const size_t layer_idx, const layer_ptr& layer )
{
    // TODO-CNN
    /*if ( layer_idx >= m_layers.size() )
    {
        std::cerr << "network_bnu_base::set_layer_ptr - cannot access layer " << layer_idx << std::endl;
        throw network_exception( "invalid layer index" );
    }

    std::cout << "network_bnu_base::set_layer_ptr - setting layer  " << layer_idx << std::endl;

    matrixF& weights = m_layers[layer_idx].weights();
    std::copy( layer.weights.get(), layer.weights.get() + layer.num_weights, &weights.data()[0] );
    vectorF& bias = m_layers[layer_idx].bias();
    std::copy( layer.bias.get(), layer.bias.get() + layer.num_bias, &bias.data()[0] );*/
}

const output_ptr lenet_bnu::output()
{
    // TODO-CNN : for now works only because last layer is one dimensional fcnn

    const vectorF& output = m_layers.back()->activations();
    output_ptr o( output.size() );
    std::copy( &output[0], &output[0] + output.size(), o.outputs.get() );

    return o;
}

// TODO-CNN : rename??
void lenet_bnu::prepare_training()
{
    for ( auto& _layer : m_layers )
    {
        _layer->prepare_training();
    }

    m_training_samples = 0;
}

void lenet_bnu::feed_forward()
{
    for ( auto& _layer : m_layers )
    {
        std::cout << "--> feed forwarding" << std::endl;
        _layer->feed_forward();
    }
}

void lenet_bnu::back_propagate()
{
    for ( auto& _layer : boost::adaptors::reverse( m_layers ) )
    {
        _layer->back_propagate();
    }

    ++m_training_samples;
}

void lenet_bnu::gradient_descent()
{
    m_optimizer->set_size( m_training_samples );

    for ( auto& _layer : m_layers )
    {
        _layer->gradient_descent( m_optimizer );
    }
}

const std::string lenet_bnu::dump_weights()
{
    return "NOT IMPLEMENTED YET";
}

const std::string lenet_bnu::dump_bias()
{
    return "NOT IMPLEMENTED YET";
}

const std::string lenet_bnu::dump_activations()
{
    return "NOT IMPLEMENTED YET";
}

}; //namespace neurocl
