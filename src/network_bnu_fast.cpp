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

#include "network_bnu_fast.h"

#ifdef __x86_64__
	#include "xmmintrin.h"
#elif __arm__
	#include <arm_neon.h>
#endif

// for tips about boost ublas matrix traversing, see:
// http://stackoverflow.com/questions/26044603/traversing-a-boostublas-matrix-using-iterators
// please note that boost ublas matrix are default row major ordered

namespace bnu = boost::numeric::ublas;

namespace neurocl {

network_bnu_fast::network_bnu_fast()
{
}

inline float _sigmoid( float x )
{
    return 1.f / ( 1.f + std::exp(-x) );
}

#ifdef __x86_64__
inline float _reduce_sum( __m128 value )
{
	/*
	 * Shuffle the input vector such that we have 1,0,3,2
	 * This is equivalent to a pairwise swap where the first
	 * two elements are swapped with the next two
	 */
	__m128 shufl = _mm_shuffle_ps(value,value, _MM_SHUFFLE(1,0,3,2));

	//Sum both values
	shufl = _mm_add_ps(value, shufl);
	//shufl = |3|2|1|0| + |1|0|3|2| = |3+1|2+0|1+3|0+2|

	/*
	 * Second shuffle 2,3,0,1
	 * This is equivalent to 1 by 1 swap between every
	 * two neighboring elements from the first swap
	 */
	__m128 shufl2 = _mm_shuffle_ps(shufl,shufl, _MM_SHUFFLE(2,3,0,1));
	//shufl2 = |2+0|3+1|0+2|1+3|

	//Sum both values
	shufl = _mm_add_ps(shufl, shufl2);
	//shufl = |3+1|2+0|1+3|0+2| + |2+0|3+1|0+2|1+3|

	//Copy the lower single-precision (32-bit) floating-point element of a to dst.
	return _mm_cvtss_f32( shufl );
	//We also could have used to extract the 0th element:
	//return _mm_extract_ps (shufl a, 0);
}
#elif __arm__
inline float _reduce_sum( float32x4_t value )
{
	float32x2_t r = vadd_f32( vget_high_f32( value ), vget_low_f32( value ) );
	return vget_lane_f32( vpadd_f32( r, r ), 0 );
	//return vgetq_lane_f32(value,0) + vgetq_lane_f32(value,1) + vgetq_lane_f32(value,2) + vgetq_lane_f32(value,3);
}
#endif

void network_bnu_fast::feed_forward()
{
    //std::cout << "network_bnu_fast::feed_forward( - " << m_layers.size() << " layers propagation" << std::endl;

    for ( size_t i=0; i<m_layers.size()-1; i++ )
    {
        vectorF& _activations1 = m_layers[i].activations();
        vectorF& _activations2 = m_layers[i+1].activations();
        matrixF& _weights = m_layers[i].weights();
        vectorF& _bias = m_layers[i].bias();

        float _temp_sum;

#ifdef __arm__

        float32x4_t _neon_temp_sum;

        // apply weights and bias, equivalent to MA + B computation
        for ( auto i = 0; i < _weights.size1(); i++ )
        {
            _temp_sum = 0.f;

            _neon_temp_sum = vdupq_n_f32( 0.f );

            for ( auto j = 0; j < _weights.size2(); j+=4 )
            {
                float32x4_t _neon_a1x4 = vld1q_f32( &_activations1[j] );
                float32x4_t _neon_wx4 = vld1q_f32( &_weights(i,j) );

                _neon_temp_sum = vmlaq_f32( _neon_temp_sum, _neon_wx4, _neon_a1x4 );
            }

			size_t tail_start = _weights.size2() - ( _weights.size2() % 4 );

			// end of the vector in non-dividable-by-4 size case
			// could be optimized more...
			for ( auto r = tail_start; r < _weights.size2(); r++ )
			{
				_temp_sum += _weights(i,r) * _activations1[r];
			}

            _activations2[i] = _sigmoid( _temp_sum + _reduce_sum( _neon_temp_sum ) + _bias[i] );
		}

#elif __x86_64__

		__m128 _mm_temp_sum;

		// apply weights and bias, equivalent to MA + B computation
		for ( auto i = 0; i < _weights.size1(); i++ )
		{
			_temp_sum = 0.f;

			_mm_temp_sum = _mm_setzero_ps();

			for ( auto j = 0; j < _weights.size2(); j+=4 )
			{
				__m128 _mm_a1x4 = _mm_load_ps( &_activations1[j] );
				__m128 _mm_wx4 = _mm_load_ps( &_weights(i,j) );

				// AVX not available on my platform :-(
				//_mm_temp_sum = _mm_fmadd_ps( _mm_wx4, _mm_a1x4, _mm_temp_sum );

				_mm_temp_sum = _mm_add_ps( _mm_temp_sum, _mm_mul_ps( _mm_wx4, _mm_a1x4 ) );
			}

			size_t tail_start = _weights.size2() - ( _weights.size2() % 4 );

			// end of the vector in non-dividable-by-4 size case
			// could be optimized more...
			for ( auto r = tail_start; r < _weights.size2(); r++ )
			{
				_temp_sum += _weights(i,r) * _activations1[r];
			}

			_activations2[i] = _sigmoid( _temp_sum + _reduce_sum( _mm_temp_sum ) + _bias[i] );
		}

#endif

    }
}

void network_bnu_fast::back_propagate()
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
        matrixF& _weights = m_layers[i].weights();
        vectorF& _activations = m_layers[i].activations();
        vectorF& _errors1 = m_layers[i].errors();
        vectorF& _errors2 = m_layers[i+1].errors();

        float _temp_sum;

        for ( auto j = 0; j < _weights.size2(); j++ )
        {
            _temp_sum = 0.f;

            for ( auto i = 0; i < _weights.size1(); i++ )
            {
                _temp_sum += _weights(i,j) * _errors2[i];
            }
            _errors1[j] = _activations[j] * ( 1.f - _activations[j] ) * _temp_sum;
        }
    }

    // Update gradients
    for ( size_t i=0; i<m_layers.size()-1; i++ )
    {
        matrixF& _w_deltas = m_layers[i].w_deltas();
        vectorF& _activations = m_layers[i].activations();
        vectorF& _errors = m_layers[i+1].errors();

#ifdef __arm__

		for ( auto k = 0; k < _w_deltas.size1(); k++ )
		{
			float32x4_t _neon_ex4 = vdupq_n_f32( _errors[k] );

			for ( auto l = 0; l < _w_deltas.size2(); l+=4 )
			{
				float* p_w_deltas = &_w_deltas(k,l);

				float32x4_t _neon_ax4 = vld1q_f32( &_activations[l] );
				float32x4_t _neon_wdx4 = vld1q_f32( p_w_deltas );

				_neon_wdx4 = vmlaq_f32( _neon_wdx4, _neon_ax4, _neon_ex4 );

				vst1q_f32( p_w_deltas, _neon_wdx4 );
			}

			size_t tail_start = _w_deltas.size2() - ( _w_deltas.size2() % 4 );

			// end of the vector in non-dividable-by-4 size case
			// could be optimized more...
			for ( auto r = tail_start; r < _w_deltas.size2(); r++ )
			{
				_w_deltas(k,r) += _errors[k] * _activations[r];
			}
		}

#elif __x86_64__

        for ( auto k = 0; k < _w_deltas.size1(); k++ )
        {
			__m128 _mm_ex4 = _mm_set1_ps( _errors[k] );

            for ( auto l = 0; l < _w_deltas.size2(); l+=4 )
            {
				float* p_w_deltas = &_w_deltas(k,l);

				__m128 _mm_ax4 = _mm_load_ps( &_activations[l] );
				__m128 _mm_wdx4 = _mm_load_ps( p_w_deltas );

				_mm_wdx4 = _mm_add_ps( _mm_wdx4, _mm_mul_ps( _mm_ax4, _mm_ex4 ) );

                _mm_store_ps( p_w_deltas, _mm_wdx4 );
            }

			size_t tail_start = _w_deltas.size2() - ( _w_deltas.size2() % 4 );

			// end of the vector in non-dividable-by-4 size case
			// could be optimized more...
			for ( auto r = tail_start; r < _w_deltas.size2(); r++ )
			{
				_w_deltas(k,r) += _errors[k] * _activations[r];
			}
        }

#endif

        m_layers[i].b_deltas() = m_layers[i].b_deltas() + m_layers[i+1].errors();
    }

    ++m_training_samples;
}

void network_bnu_fast::gradient_descent()
{
    //std::cout << "network_bnu::gradient_descent - updating after " << m_training_samples << " backpropagations" << std::endl;

    float invm = 1.f / static_cast<float>( m_training_samples );

    for ( size_t i=0; i<m_layers.size()-1; i++ ) // avoid output layer
    {
        m_layers[i].weights() -= m_learning_rate * ( ( invm * m_layers[i].w_deltas() ) + ( m_weight_decay * m_layers[i].weights() ) );
        m_layers[i].bias() -= m_learning_rate * ( invm * m_layers[i].b_deltas() );
    }
}

}; //namespace neurocl
