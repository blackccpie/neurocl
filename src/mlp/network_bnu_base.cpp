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

#include "network_bnu_base.h"

#include "common/network_config.h"
#include "common/network_exception.h"
#include "common/network_utils.h"

#include <boost/foreach.hpp>
#include <boost/optional.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>

namespace bnu = boost::numeric::ublas;

namespace neurocl { namespace mlp {

const std::string dump_mat( const matrixF& mat, boost::optional<std::string> label = boost::none )
{
    std::string separator;
    std::stringstream ss;
    ss << ( label ? label.get() : "" ) << std::endl;
    for( matrixF::const_iterator1 it1 = mat.begin1(); it1 != mat.end1(); ++it1 )
    {
        for( matrixF::const_iterator2 it2 = it1.begin(); it2 !=it1.end(); ++it2 )
        {
            ss << separator << *it2;
            separator = " ";
        }
        separator = "";
        ss << std::endl;
    }
    return ss.str();
}

const std::string dump_vec( const vectorF& vec, boost::optional<std::string> label = boost::none )
{
    std::string separator;
    std::stringstream ss;
    ss << ( label ? label.get() : "" ) << std::endl;
    for( auto val : vec )
    {
            ss << separator << val;
            separator = " ";
    }
    ss << std::endl;
    return ss.str();
}

template<class T>
void random_normal_init( T& container, const float stddev = 1.f )
{
    utils::rand_gaussian_generator rgg( 0.f, stddev );

    BOOST_FOREACH( float& element, container.data() )
    {
        element = rgg();
    }
}

layer_bnu::layer_bnu()
{
}

// WARNING : size is the square side size
void layer_bnu::populate( const layer_size& cur_layer_size, const layer_size& next_layer_size )
{
    //LOGGER(info) << "layer_bnu::populate - populating layer of size " << cur_layer_size << " (next size is " << next_layer_size << ")" << std::endl;

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

const std::string layer_bnu::dump_weights() const
{
    return dump_mat( m_output_weights );
}

const std::string layer_bnu::dump_bias() const
{
    return dump_vec( m_bias );
}

const std::string layer_bnu::dump_activations() const
{
    return dump_vec( m_activations );
}

network_bnu_base::network_bnu_base() : m_learning_rate( 3.0f/*0.01f*/ ), m_weight_decay( 0.0f ), m_training_samples( 0 )
{
    const network_config& nc = network_config::instance();
    nc.update_optional( "learning_rate", m_learning_rate );
}

void network_bnu_base::set_input(  const size_t& in_size, const float* in )
{
    if ( in_size > m_layers[0].activations().size() )
        throw network_exception( "sample size exceeds allocated layer size!" );

    //LOGGER(info) << "network_bnu::set_input - input (" << in << ") size = " << in_size << std::endl;

    vectorF& input_activations = m_layers[0].activations();
    std::copy( in, in + in_size, input_activations.begin() );
}

void network_bnu_base::set_output( const size_t& out_size, const float* out )
{
    if ( out_size > m_training_output.size() )
        throw network_exception( "output size exceeds allocated layer size!" );

    //LOGGER(info) << "network_bnu::set_output - output (" << out << ") size = " << out_size << std::endl;

    std::copy( out, out + out_size, m_training_output.begin() );
}

void network_bnu_base::add_layers_2d( const std::vector<layer_size>& layer_sizes )
{
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
    }
}

const layer_ptr network_bnu_base::get_layer_ptr( const size_t layer_idx )
{
    if ( layer_idx >= m_layers.size() )
    {
        LOGGER(error) << "network_bnu_base::get_layer_ptr - cannot access layer " << layer_idx << std::endl;
        throw network_exception( "invalid layer index" );
    }

    matrixF& weights = m_layers[layer_idx].weights();
    vectorF& bias = m_layers[layer_idx].bias();
    layer_ptr l( weights.size1() * weights.size2(), bias.size() );
    std::copy( &weights.data()[0], &weights.data()[0] + ( weights.size1() * weights.size2() ), l.weights.get() );
    std::copy( &bias[0], &bias[0] + bias.size(), l.bias.get() );

    return l;
}

void network_bnu_base::set_layer_ptr( const size_t layer_idx, const layer_ptr& layer )
{
    if ( layer_idx >= m_layers.size() )
    {
        LOGGER(error) << "network_bnu_base::set_layer_ptr - cannot access layer " << layer_idx << std::endl;
        throw network_exception( "invalid layer index" );
    }

    LOGGER(info) << "network_bnu_base::set_layer_ptr - setting layer  " << layer_idx << std::endl;

    matrixF& weights = m_layers[layer_idx].weights();
    std::copy( layer.weights.get(), layer.weights.get() + layer.num_weights, &weights.data()[0] );
    vectorF& bias = m_layers[layer_idx].bias();
    std::copy( layer.bias.get(), layer.bias.get() + layer.num_bias, &bias.data()[0] );
}

const output_ptr network_bnu_base::output()
{
    vectorF& output = m_layers.back().activations();
    output_ptr o( output.size() );
    std::copy( &output[0], &output[0] + output.size(), o.outputs.get() );

    return o;
}

void network_bnu_base::prepare_training()
{
    // Clear gradients
    for ( size_t i=0; i<m_layers.size()-1; i++ )
    {
        m_layers[i].w_deltas().clear();
        m_layers[i].b_deltas().clear();
    }

    m_training_samples = 0;
}

const std::string network_bnu_base::dump_weights()
{
    std::stringstream ss;
    ss << "*************************************************" << std::endl;
    BOOST_FOREACH( const layer_bnu& layer, m_layers )
    {
        ss << layer.dump_weights();
        ss << "-------------------------------------------------" << std::endl;
    }
    ss << "*************************************************" << std::endl;
    return ss.str();
}

const std::string network_bnu_base::dump_bias()
{
    std::stringstream ss;
    ss << "*************************************************" << std::endl;
    BOOST_FOREACH( const layer_bnu& layer, m_layers )
    {
        ss << layer.dump_bias();
        ss << "-------------------------------------------------" << std::endl;
    }
    ss << "*************************************************" << std::endl;
    return ss.str();
}

const std::string network_bnu_base::dump_activations()
{
    std::stringstream ss;
    ss << "*************************************************" << std::endl;
    BOOST_FOREACH( const layer_bnu& layer, m_layers )
    {
        ss << layer.dump_activations();
        ss << "-------------------------------------------------" << std::endl;
    }
    ss << "*************************************************" << std::endl;
    return ss.str();
}

} /*namespace neurocl*/ } /*namespace mlp*/
