#include "network.h"

namespace neurocl {

VEX_CONSTANT(_zero, 0.f);
VEX_CONSTANT(_one, 1.f);

vex::Context g_ctx( vex::Filter::GPU && vex::Filter::Count(1) );

layer::layer()
{
}

// WARNING : size is the square side size
void layer::populate( const size_t& size, const size_t& next_layer_size )
{
    std::cout << "populating layer of size " << size << " (next size is " << next_layer_size << ")" << std::endl;

    if ( next_layer_size ) // non-output layer
    {
        m_weights_size = std::make_pair( next_layer_size, size * size );
        m_output_weights = vex::vector<float>( g_ctx, next_layer_size * next_layer_size * size * size );
        m_output_weights = _one();
        m_deltas_weight = vex::vector<float>( g_ctx, next_layer_size * next_layer_size * size * size );
        m_deltas_weight - _zero();
    }

    m_bias = vex::vector<float>( g_ctx, size * size );
    m_bias = _zero();
    m_deltas_bias = vex::vector<float>( g_ctx, size * size );
    m_deltas_bias = _zero();

    m_activations = vex::vector<float>( g_ctx, size * size );
    m_activations = _zero();
    m_errors = vex::vector<float>( g_ctx, size * size );
    m_errors = _zero();
}

network::network() : m_learning_rate( 0.01f ), m_weight_decay( 0.1f /*TBC*/)
{
    if ( !g_ctx ) throw std::runtime_error( "No devices available." );

    // Print out list of selected devices:
    std::cout << g_ctx << std::endl;
}

void network::set_training_sample(  const size_t& isample_size, const float* isample,
                                const size_t& osample_size, const float* osample )
{
    // TODO manage case where sample_size exceeds layer size

    vex::copy( isample, isample + isample_size, m_layers[0].activations().begin() );
    vex::copy( osample, osample + osample_size, m_training_output.begin() );
}

void network::add_layers_2d( const std::vector<size_t>& layer_sizes )
{
    m_layers.resize( layer_sizes.size() );

    // Last layer should be output layer
    size_t _last_size = layer_sizes.back();
    m_layers.back().populate( _last_size, 0 );

    // Initialize training output
    m_training_output = vex::vector<float>( g_ctx, _last_size * _last_size );

    // Populate all but input layer
    for ( int idx=layer_sizes.size()-2; idx>=0; idx-- )
    {
        const size_t& _size = layer_sizes[idx];
        const size_t& _next_layer_size = layer_sizes[idx+1];
        m_layers[idx].populate( _size, _next_layer_size );
    }
}

void network::feed_forward()
{
    std::cout << m_layers.size() << " layers propagation" << std::endl;

    for ( size_t i=0; i<m_layers.size()-1; i++ )
    {
        std::cout << "feed_forward layer " << i << std::endl;

        size_t n = m_layers[i].w_size().first;
        size_t m = m_layers[i].w_size().second;

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
            + m_layers[i+1].bias() ) ) // watch out the index +1 compared to stanford
        );
    }
}

const float network::output()
{
    return m_layers.back().activations()[0];
}

void network::_back_propagate()
{
    // PREREQUISITE : FEED FORWARD PASS

    // Output layer error vector
    layer& output_layer = m_layers.back();
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
        m_layers[i].b_deltas() += m_layers[i].errors(); // watch out the index compared to stanford
    }

}

void network::gradient_descent()
{
    _back_propagate();
    _gradient_descent();
}

void network::_gradient_descent()
{
    for ( size_t i=0; i<m_layers.size()-1; i++ ) // avoid output layer
    {
        float m = static_cast<float>( m_layers[i].w_size().second );

        m_layers[i].weights() -= ( m_learning_rate * ( m_layers[i].w_deltas() / m ) + ( m_weight_decay * m_layers[i].weights() ) );
        m_layers[i].bias() -= m_learning_rate * ( m_layers[i].b_deltas() / m );
    }
}

}; //namespace neurocl
