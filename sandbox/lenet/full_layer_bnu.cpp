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

#include "full_layer_bnu.h"
#include "network_utils.h"

namespace bnu = boost::numeric::ublas;

namespace neurocl {

inline float sigmoid( float x )
{
    return 1.f / ( 1.f + std::exp(-x) );
}

template<class T>
void random_normal_init( T& container, const float stddev = 1.f )
{
    utils::rand_gaussian_generator rgg( 0.f, stddev );

    for( auto& element : container.data() )
    {
        element = rgg();
    }
}

full_layer_bnu::full_layer_bnu()
{
}

// WARNING : size is the square side size
void full_layer_bnu::populate(  layer_bnu* prev_layer,
                                const layer_size& lsize )
{
    std::cout << "populating full layer" << std::endl;

    m_prev_layer = prev_layer;

    if ( m_prev_layer ) // create a separate layer type for input???
    {
        m_weights = matrixF( lsize.size(), m_prev_layer->size() );
        // cf. http://neuralnetworksanddeeplearning.com/chap3.html#weight_initialization
        random_normal_init( m_weights, 1.f / std::sqrt( m_prev_layer->size() ) );
        m_deltas_weight = matrixF( lsize.size(), m_prev_layer->size() );
        m_deltas_weight.clear();

        m_bias = vectorF( lsize.size() );
        random_normal_init( m_bias, 1.f );
        m_deltas_bias = vectorF( lsize.size() );
        m_deltas_bias.clear();
    }

    m_activations = vectorF( lsize.size() );
    m_activations.clear();
    m_errors = vectorF( lsize.size() ); // not needed for input layer...?
    m_errors.clear();
}

void full_layer_bnu::feed_forward()
{
    if ( m_prev_layer->has_feature_maps() )
    {
        vectorF reconstructed_vector( m_prev_layer->width() * m_prev_layer->height() * m_prev_layer->depth() );

        for ( auto i=0; i<m_prev_layer->depth(); i++ )
        {
            auto& feature_map = m_prev_layer->feature_map(i).data();

            std::copy( feature_map.begin(), feature_map.end(),
                reconstructed_vector.data().begin() + ( i * m_prev_layer->width() * m_prev_layer->height() ) );
        }
    }
    else
    {
        const vectorF& prev_activations = m_prev_layer->activations();

        // apply weights and bias
        m_activations = bnu::prod( m_weights, prev_activations )
            + m_bias;
    }

    // apply sigmoid function
    std::for_each( m_activations.data().begin(), m_activations.data().end(), std::ptr_fun( sigmoid ) );
}

void full_layer_bnu::prepare_training()
{
    m_deltas_weight.clear();
    m_deltas_bias.clear();
}

void full_layer_bnu::back_propagate()
{
    // Compute errors
    for ( auto i=0; i<m_prev_layer->depth(); i++ )
    {
        // TODO-CNN
    	/*m_prev_layer->errors_maps(i) = bnu::element_prod(
        	bnu::element_prod(  m_activations,
                            	( bnu::scalar_vector<float>( m_activations.size(), 1.f ) - m_activations ) ),
        	bnu::prod( bnu::trans( m_weights ), m_errors ) );*/
    }

    // Compute gradients

    /*
    m_layers[i].w_deltas() = m_layers[i].w_deltas() + bnu::outer_prod( m_layers[i+1].errors(), m_layers[i].activations() );
    m_layers[i].b_deltas() = m_layers[i].b_deltas() + m_layers[i+1].errors();
    */
}

void full_layer_bnu::gradient_descent( boost::shared_ptr<optimizer> optimizer )
{
    // Update weights and bias according to gradients

    /*auto invm = 1.f / static_cast<float>( m_training_samples );

    m_layers[i].weights() -= m_learning_rate * ( ( invm * m_layers[i].w_deltas() ) + ( m_weight_decay * m_layers[i].weights() ) );
    m_layers[i].bias() -= m_learning_rate * ( invm * m_layers[i].b_deltas() );*/
}

}; //namespace neurocl
