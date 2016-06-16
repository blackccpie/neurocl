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

#include <boost/foreach.hpp>
#include <boost/optional.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>

namespace bnu = boost::numeric::ublas;

namespace neurocl {

template<class T>
void random_normal_init( T& container, const float stddev = 1.f )
{
    utils::rand_gaussian_generator rgg( 0.f, stddev );

    BOOST_FOREACH( float& element, container.data() )
    {
        element = rgg();
    }
}

full_layer_bnu::full_layer_bnu()
{
}

// WARNING : size is the square side size
void full_layer_bnu::populate( const layer_size& cur_layer_size, const layer_size& next_layer_size )
{
    //std::cout << "populating layer of size " << cur_layer_size << " (next size is " << next_layer_size << ")" << std::endl;

    if ( next_layer_size.size() ) // non-output layer
    {
        m_output_weights = matrixF( next_layer_size.size(), cur_layer_size.size() );
        // cf. http://neuralnetworksanddeeplearning.com/chap3.html#weight_initialization
        random_normal_init( m_output_weights, 1.f / std::sqrt( cur_layer_size.size() ) );
        m_deltas_weight = matrixF( next_layer_size.size(), cur_layer_size.size() );
        m_deltas_weight.clear();

        m_bias = vectorF( next_layer_size.size() );
        random_normal_init( m_bias, 1.f );
        m_deltas_bias = vectorF( next_layer_size.size() );
        m_deltas_bias.clear();
    }

    m_activations = vectorF( cur_layer_size.size() );
    m_activations.clear();
    m_errors = vectorF( cur_layer_size.size() ); // not needed for input layer...?
    m_errors.clear();
}

void full_layer_bnu::feed_forward( const layer_iface* prev_layer )
{
    // NOT IMPLEMENTED YET
}

const std::string full_layer_bnu::dump_weights() const
{
    return "NOT IMPLEMENTED YET";
}

const std::string full_layer_bnu::dump_bias() const
{
    return "NOT IMPLEMENTED YET";
}

const std::string full_layer_bnu::dump_activations() const
{
    return "NOT IMPLEMENTED YET";
}

conv_layer_bnu::conv_layer_bnu()
{
}

void conv_layer_bnu::set_filter_size( const size_t filter_size, const size_t filter_stride )
{
    m_filter_size = filter_size;
    m_filter_stride = filter_stride;
}

void conv_layer_bnu::populate( const size_t width, const size_t height, const size_t depth )
{
    m_filters = boost::shared_array<matrixF>( new matrixF[depth] );
    m_feature_maps = boost::shared_array<matrixF>( new matrixF[depth] );
    for ( size_t i = 0; i<depth; i++ )
    {
        m_feature_maps[i] = matrixF( width, height );
        m_filters[i] = matrixF( width, height );
        random_normal_init( m_filters[i], 1.f / std::sqrt( m_filter_size * m_filter_size ) );
    }
}

void conv_layer_bnu::feed_forward( const layer_iface* prev_layer )
{
    // NOT IMPLEMENTED YET
}

pool_layer_bnu::pool_layer_bnu()
{

}

void pool_layer_bnu::populate( const size_t width, const size_t height, const size_t depth )
{
    m_feature_maps = boost::shared_array<matrixF>( new matrixF[depth] );
    for ( size_t i = 0; i<depth; i++ )
    {
        m_feature_maps[i] = matrixF( width, height );
        random_normal_init( m_feature_maps[i], 1.f / std::sqrt( width * height ) );
    }
}

void pool_layer_bnu::feed_forward( const layer_iface* prev_layer )
{
    // NOT IMPLEMENTED YET
}

lenet_bnu::lenet_bnu() : m_learning_rate( 3.0f/*0.01f*/ ), m_weight_decay( 0.0f ), m_training_samples( 0 )
{
    const network_config& nc = network_config::instance();
    nc.update_optional( "learning_rate", m_learning_rate );
}

void lenet_bnu::set_input(  const size_t& in_size, const float* in )
{
    // TODO-CNN
    /*if ( in_size > m_layers[0].activations().size() )
        throw network_exception( "sample size exceeds allocated layer size!" );

    //std::cout << "network_bnu::set_input - input (" << in << ") size = " << in_size << std::endl;

    vectorF& input_activations = m_layers[0].activations();
    std::copy( in, in + in_size, input_activations.begin() );*/
}

void lenet_bnu::set_output( const size_t& out_size, const float* out )
{
    if ( out_size > m_training_output.size() )
        throw network_exception( "output size exceeds allocated layer size!" );

    //std::cout << "network_bnu::set_output - output (" << out << ") size = " << out_size << std::endl;

    std::copy( out, out + out_size, m_training_output.begin() );
}

void lenet_bnu::add_layers_2d( const std::vector<layer_size>& layer_sizes )
{
    m_layer_input.populate( layer_size( 32, 32 ), layer_size( /*TODO*/0, 0 ) );
    m_layer_c1.set_filter_size( 5 );
    m_layer_c1.populate( 28, 28, 6 );               //conv
    m_layer_s2.populate( 14, 14, 6 );               //pool
    m_layer_c3.set_filter_size( 6 );
    m_layer_c3.populate( 10, 10, 16 );              //conv
    m_layer_s4.populate( 5, 5, 16 );                //pool
    //TBC m_layer_c5.populate( 120 );    //conv
    m_layer_f6.populate( layer_size( 12, 7 ), layer_size( 10, 1 ) );     //full
    m_layer_output.populate( layer_size( 10, 1 ), layer_size( 0, 0 ) );

    m_layers = { &m_layer_c1 };

    // TODO-CNN : STUBBED FOR NOW
    /*
    m_layers.resize( layer_sizes.size() );

    // Last layer should be output layer
    const layer_size& _last_size = layer_sizes.back();
    m_layers.back().populate( _last_size, layer_size( 0, 0 ) );

    // Initialize training output
    m_training_output = vectorF( _last_size.size() );

    // Populate all but input layer
    for ( int idx=layer_sizes.size()-2; idx>=0; idx-- )
    {
        const layer_size& _size = layer_sizes[idx];
        const layer_size& _next_layer_size = layer_sizes[idx+1];
        m_layers[idx].populate( _size, _next_layer_size );
    }*/
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
    // TODO-CNN
    /*vectorF& output = m_layers.back().activations();
    output_ptr o( output.size() );
    std::copy( &output[0], &output[0] + output.size(), o.outputs.get() );

    return o;*/
    return output_ptr(0);
}

void lenet_bnu::prepare_training()
{
    // TODO-CNN
    /*
    // Clear gradients
    for ( size_t i=0; i<m_layers.size()-1; i++ )
    {
        m_layers[i].w_deltas().clear();
        m_layers[i].b_deltas().clear();
    }

    m_training_samples = 0;
    */
}

void lenet_bnu::feed_forward()
{
    /*for ( size_t i=0; i<m_layers.size()-1; i++ )
    {
        vectorF& _activations = m_layers[i+1].activations();

        // apply weights and bias
        _activations = bnu::prod( m_layers[i].weights(), m_layers[i].activations() )
            + m_layers[i].bias();

        // apply sigmoid function
        std::transform( _activations.data().begin(), _activations.data().end(),
            _activations.data().begin(), std::ptr_fun( sigmoid ) );
    }*/
}

void lenet_bnu::back_propagate()
{

}

void lenet_bnu::gradient_descent()
{

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
