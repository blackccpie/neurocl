#include "network_bnu.h"

namespace bnu = boost::numeric::ublas;

namespace neurocl {

layer_bnu::layer_bnu()
{
}

// WARNING : size is the square side size
void layer_bnu::populate( const size_t& size, const size_t& next_layer_size )
{
    std::cout << "populating layer of size " << size << " (next size is " << next_layer_size << ")" << std::endl;

    if ( next_layer_size ) // non-output layer
    {
        m_output_weights = matrixF( next_layer_size * next_layer_size, size * size );
        std::fill( m_output_weights.data().begin(), m_output_weights.data().end(), 1.0f );
        m_deltas_weight = matrixF( next_layer_size * next_layer_size, size * size );
        m_deltas_weight.clear();
    }

    m_bias = vectorF( size * size );
    m_bias.clear();
    m_deltas_bias = vectorF( size * size );
    m_deltas_bias.clear();

    m_activations = vectorF( size * size );
    m_activations.clear();
    m_errors = vectorF( size * size );
    m_errors.clear();
}

network_bnu::network_bnu() : m_learning_rate( 0.01f ), m_weight_decay( 0.1f /*TBC*/)
{
}

void network_bnu::set_training_sample(  const size_t& isample_size, const float* isample,
                                        const size_t& osample_size, const float* osample )
{
    // TODO : manage case where sample_size exceeds layer size

    vectorF& input_activations = m_layers[0].activations();
    std::copy( isample, isample + isample_size, input_activations.begin() );
    std::copy( osample, osample + osample_size, m_training_output.begin() );
}

void network_bnu::add_layers_2d( const std::vector<size_t>& layer_sizes )
{
    m_layers.resize( layer_sizes.size() );

    // Last layer should be output layer
    size_t _last_size = layer_sizes.back();
    m_layers.back().populate( _last_size, 0 );

    // Initialize training output
    m_training_output = vectorF( _last_size * _last_size );

    // Populate all but input layer
    for ( int idx=layer_sizes.size()-2; idx>=0; idx-- )
    {
        const size_t& _size = layer_sizes[idx];
        const size_t& _next_layer_size = layer_sizes[idx+1];
        m_layers[idx].populate( _size, _next_layer_size );
    }
}

float sigmoid(float x)
{
  return 1.f / ( 1.f + std::exp(-x) );
}

void network_bnu::feed_forward()
{
    std::cout << m_layers.size() << " layers propagation" << std::endl;

    for ( size_t i=0; i<m_layers.size()-1; i++ )
    {
        std::cout << "feed_forward layer " << i << std::endl;

        vectorF& _activations = m_layers[i+1].activations();

        // apply weights and bias
        _activations = bnu::prod( m_layers[i].weights(), m_layers[i].activations() )
            + m_layers[i+1].bias(); // watch out the index +1 compared to stanford

        // apply sigmoid function
        std::transform( _activations.data().begin(), _activations.data().end(),
            _activations.data().begin(), std::ptr_fun( sigmoid ) );
    }
}

const float network_bnu::output()
{
    return m_layers.back().activations()[0];
}

void network_bnu::_back_propagate()
{
    // PREREQUISITE : FEED FORWARD PASS

    // Output layer error vector
    layer_bnu& output_layer = m_layers.back();
    output_layer.errors() = bnu::element_prod(
            bnu::element_prod(  output_layer.activations(),
                                ( bnu::unit_vector<float>( output_layer.activations().size() ) - output_layer.activations() ) ),
            ( output_layer.activations() - m_training_output ) );

    // Hidden layers error vectors
    for ( size_t i=m_layers.size()-2; i>0; i-- )
    {
        m_layers[i].errors() = bnu::element_prod(
            bnu::element_prod(  m_layers[i].activations(),
                                ( bnu::unit_vector<float>( m_layers[i].activations().size() ) - m_layers[i].activations() ) ),
            bnu::prod( bnu::trans( m_layers[i].weights() ), m_layers[i+1].errors() ) );
    }

    // Update gradients
    for ( size_t i=0; i<m_layers.size()-1; i++ )
    {
        m_layers[i].w_deltas() = m_layers[i].w_deltas() + bnu::outer_prod( m_layers[i+1].errors(), m_layers[i].activations() );
        m_layers[i].b_deltas() = m_layers[i].b_deltas()+ m_layers[i].errors(); // watch out the index compared to stanford
    }
}

void network_bnu::gradient_descent()
{
    _back_propagate();
    _gradient_descent();
}

void network_bnu::_gradient_descent()
{
    for ( size_t i=0; i<m_layers.size()-1; i++ ) // avoid output layer
    {
        float m = static_cast<float>( m_layers[i].weights().size2() );

        m_layers[i].weights() -= ( m_learning_rate * ( m_layers[i].w_deltas() / m ) + ( m_weight_decay * m_layers[i].weights() ) );
        m_layers[i].bias() -= m_learning_rate * ( m_layers[i].b_deltas() / m );
    }
}

}; //namespace neurocl
