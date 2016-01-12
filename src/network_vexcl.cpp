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
#include "network_exception.h"
#include "network_utils.h"

#include <boost/shared_array.hpp>

namespace neurocl {

VEX_CONSTANT(_zero, 0.f);
VEX_CONSTANT(_one, 1.f);

vex::Context g_ctx( vex::Filter::GPU && vex::Filter::Count(1) );

void random_normal_init( vex::vector<float>& container, const float stddev = 1.f )
{
    utils::rand_gaussian_generator rgg( 0.f, stddev );

    boost::shared_array<float> arand( new float[container.size()] );

    for ( size_t i=0; i<container.size(); i++ )
        arand[i] = rgg();

    vex::copy( arand.get(), arand.get()+container.size(), container.begin() );
}

layer_vexcl::layer_vexcl()
{
}

// WARNING : size is the square side size
void layer_vexcl::populate( const layer_size& cur_layer_size, const layer_size& next_layer_size )
{
    //std::cout << "populating layer of size " << cur_layer_size << " (next size is " << next_layer_size << ")" << std::endl;

    if ( next_layer_size.size() ) // non-output layer
    {
        m_weights_size = std::make_pair( next_layer_size.size(), cur_layer_size.size() );
        m_output_weights = vex::vector<float>( g_ctx, next_layer_size.size() * cur_layer_size.size() );
        // cf. http://neuralnetworksanddeeplearning.com/chap3.html#weight_initialization
        random_normal_init( m_output_weights, 1.f / std::sqrt( cur_layer_size.size() ) );
        m_deltas_weight = vex::vector<float>( g_ctx, next_layer_size.size() * cur_layer_size.size() );
        m_deltas_weight - _zero();

        m_bias = vex::vector<float>( g_ctx, next_layer_size.size() );
        random_normal_init( m_bias, 1.f );
        m_deltas_bias = vex::vector<float>( g_ctx, next_layer_size.size() );
        m_deltas_bias = _zero();
    }

    m_activations = vex::vector<float>( g_ctx, cur_layer_size.size() );
    m_activations = _zero();
    m_errors = vex::vector<float>( g_ctx, cur_layer_size.size() );
    m_errors = _zero();
}

network_vexcl::network_vexcl() : m_learning_rate( 3.0f/*0.01f*/ ), m_weight_decay( 0.01f ), m_training_samples( 0 )
{
    if ( !g_ctx ) throw std::runtime_error( "No devices available." );

    // Print out list of selected devices:
    std::cout << g_ctx << std::endl;
}

void network_vexcl::set_input_sample( const size_t& isample_size, const float* isample,
                                const size_t& osample_size, const float* osample )
{
    // TODO manage case where sample_size exceeds layer size

    vex::copy( isample, isample + isample_size, m_layers[0].activations().begin() );
    vex::copy( osample, osample + osample_size, m_training_output.begin() );
}

void network_vexcl::add_layers_2d( const std::vector<layer_size>& layer_sizes )
{
    m_layers.resize( layer_sizes.size() );

    // Last layer should be output layer
    const layer_size& _last_size = layer_sizes.back();
    m_layers.back().populate( _last_size, layer_size( 0, 0 ) );

    // Initialize training output
    m_training_output = vex::vector<float>( g_ctx, _last_size.size() );

    // Populate all but input layer
    for ( int idx=layer_sizes.size()-2; idx>=0; idx-- )
    {
        const layer_size& _size = layer_sizes[idx];
        const layer_size& _next_layer_size = layer_sizes[idx+1];
        m_layers[idx].populate( _size, _next_layer_size );
    }
}

void network_vexcl::feed_forward()
{
    std::cout << m_layers.size() << " layers propagation" << std::endl;

    for ( size_t i=0; i<m_layers.size()-1; i++ )
    {
        std::cout << "feed_forward layer " << i << std::endl;

        const size_t n = m_layers[i].w_size().first;
        const size_t m = m_layers[i].w_size().second;

        m_layers[i+1].activations() = _one() / ( _one() + exp(
            -( vex::reduce<vex::SUM>(
                vex::extents[n][m],     // Shape of the expression to reduce,
                m_layers[i].weights()
                *
                vex::reshape(
                    m_layers[i].activations(),
                    vex::extents[n][m], // (We need an n x m matrix...
                    vex::extents[1]     // ... but we only have vector of size m).
                ),                      // the expression,
                1                       // and the dimension to reduce along.
            )
            + m_layers[i].bias() ) )
        );
    }
}

const layer_ptr network_vexcl::get_layer_ptr( const size_t layer_idx )
{
    if ( layer_idx >= m_layers.size() )
    {
        std::cerr << "network_vexcl::get_layer_ptr - cannot access layer " << layer_idx << std::endl;
        throw network_exception( "invalid layer index" );
    }

    // NOT IMPLEMENTED YET;
    return layer_ptr( 0, 0, 0, 0 );
}

void network_vexcl::set_layer_ptr( const size_t layer_idx, const layer_ptr& layer )
{
    if ( layer_idx >= m_layers.size() )
    {
        std::cerr << "network_vexcl::set_layer_ptr - cannot access layer " << layer_idx << std::endl;
        throw network_exception( "invalid layer index" );
    }

    // NOT IMPLEMENTED YET;
}

const float network_vexcl::output()
{
    // very slow for a GPU backend!!!
    return m_layers.back().activations()[0];
}

const float network_vexcl::error()
{
    return m_layers.back().errors()[0];
}

void network_vexcl::prepare_training()
{
    // Clear gradients
    for ( size_t i=0; i<m_layers.size(); i++ )
    {
        m_layers[i].w_deltas() = _zero();
        m_layers[i].b_deltas() = _zero();
    }

    m_training_samples = 0;
}

void network_vexcl::back_propagate()
{
    // PREREQUISITE : FEED FORWARD PASS

    // Output layer error vector
    layer_vexcl& output_layer = m_layers.back();
    output_layer.errors() = output_layer.activations() * ( _one() - output_layer.activations() )
        * ( output_layer.activations() - vex::constant(m_training_output) );

    // Hidden layers error vectors
    for ( size_t i=m_layers.size()-2; i>0; i-- )
    {
        size_t n = m_layers[i].w_size().first;
        size_t m = m_layers[i].w_size().second;

        m_layers[i].errors() = m_layers[i].activations() * ( _one() - m_layers[i].activations() )
            *
            vex::reduce<vex::SUM>(
                vex::extents[m][n],     // Shape of the expression to reduce,
                vex::reshape(
                    m_layers[i].weights(),
                    vex::extents[m][n], // new shape
                    vex::extents[1][0]  // m_layers[i].weights is shaped as [n][m]
                )
                *
                vex::reshape(
                    m_layers[i+1].errors(),
                    vex::extents[m][n], // (We need an m x n matrix...
                    vex::extents[1]     // ... but we only have vector of size m).
                ),                      // the expression,
                1                       // and the dimension to reduce along.
            );
    }

    // Update gradients
    for ( size_t i=0; i<m_layers.size()-1; i++ )
    {
        size_t n = m_layers[i].w_size().first;
        size_t m = m_layers[i].w_size().second;

        m_layers[i].w_deltas() += vex::reshape( m_layers[i+1].errors(), vex::extents[n][1], vex::extents[0][1] )
            * vex::reshape( m_layers[i].activations(), vex::extents[1][m], vex::extents[1][0] );
        m_layers[i].b_deltas() += m_layers[i+1].errors();
    }

    ++m_training_samples;
}

void network_vexcl::update_params()
{
    std::cout << "network_bnu::update_params - updating after " << m_training_samples << " backpropagations" << std::endl;

    float invm = 1.f / static_cast<float>( m_training_samples );

    for ( size_t i=0; i<m_layers.size()-1; i++ ) // avoid output layer
    {
        m_layers[i].weights() -= m_learning_rate * ( ( invm * m_layers[i].w_deltas() ) + ( m_weight_decay * m_layers[i].weights() ) );
        m_layers[i].bias() -= m_learning_rate * ( invm * m_layers[i].b_deltas() );
    }
}

const std::string network_vexcl::dump_weights()
{
    return "WEIGHTS DUMPING NOT IMPLEMENTED YET";
}

const std::string network_vexcl::dump_activations()
{
    return "ACTIVATIONS DUMPING NOT IMPLEMENTED YET";
}

}; //namespace neurocl