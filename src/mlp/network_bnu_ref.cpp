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

#include "network_bnu_ref.h"

namespace bnu = boost::numeric::ublas;

namespace neurocl { namespace mlp {

network_bnu_ref::network_bnu_ref()
{
}

float sigmoid( float x )
{
    return 1.f / ( 1.f + std::exp(-x) );
}

void network_bnu_ref::feed_forward()
{
    //LOGGER(info) << "network_bnu_ref::feed_forward - " << m_layers.size() << " layers propagation" << std::endl;

    for ( size_t i=0; i<m_layers.size()-1; i++ )
    {
        vectorF& _activations = m_layers[i+1].activations();

        // apply weights and bias
        _activations = bnu::prod( m_layers[i].weights(), m_layers[i].activations() )
            + m_layers[i].bias();

        // apply sigmoid function
        std::transform( _activations.data().begin(), _activations.data().end(),
            _activations.data().begin(), std::ptr_fun( sigmoid ) );
    }
}

void network_bnu_ref::back_propagate()
{
    // PREREQUISITE : FEED FORWARD PASS

    // Output layer error vector
    layer_bnu& output_layer = m_layers.back();
    output_layer.errors() = bnu::element_prod(
            bnu::element_prod(  output_layer.activations(),
                                ( bnu::scalar_vector<float>( output_layer.activations().size(), 1.f ) - output_layer.activations() ) ),
            ( output_layer.activations() - m_training_output ) );

    // Hidden layers error vectors
    for ( size_t i=m_layers.size()-2; i>0; i-- )
    {
        m_layers[i].errors() = bnu::element_prod(
            bnu::element_prod(  m_layers[i].activations(),
                                ( bnu::scalar_vector<float>( m_layers[i].activations().size(), 1.f ) - m_layers[i].activations() ) ),
            bnu::prod( bnu::trans( m_layers[i].weights() ), m_layers[i+1].errors() ) );
    }

    // Update gradients
    for ( size_t i=0; i<m_layers.size()-1; i++ )
    {
        m_layers[i].w_deltas() = m_layers[i].w_deltas() + bnu::outer_prod( m_layers[i+1].errors(), m_layers[i].activations() );
        m_layers[i].b_deltas() = m_layers[i].b_deltas() + m_layers[i+1].errors();
    }

    ++m_training_samples;
}

void network_bnu_ref::gradient_descent()
{
    //LOGGER(info) << "network_bnu_ref::gradient_descent - updating after " << m_training_samples << " backpropagations" << std::endl;

    auto invm = 1.f / static_cast<float>( m_training_samples );

    for ( size_t i=0; i<m_layers.size()-1; i++ ) // avoid output layer
    {
        m_layers[i].weights() -= m_learning_rate * ( ( invm * m_layers[i].w_deltas() ) + ( m_weight_decay * m_layers[i].weights() ) );
        m_layers[i].bias() -= m_learning_rate * ( invm * m_layers[i].b_deltas() );
    }
}

} /*namespace neurocl*/ } /*namespace mlp*/
