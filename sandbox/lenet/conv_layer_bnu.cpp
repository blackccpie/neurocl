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

#include "conv_layer_bnu.h"
#include "network_utils.h"
#include "network_exception.h"

#include <boost/numeric/ublas/matrix_proxy.hpp>

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

conv_layer_bnu::conv_layer_bnu() : m_filter_size( 0 ), m_filter_stride( 0 )
{
}

void conv_layer_bnu::set_filter_size( const size_t filter_size, const size_t filter_stride )
{
    m_filter_size = filter_size;
    m_filter_stride = filter_stride;
}

void conv_layer_bnu::populate(  layer_bnu* prev_layer,
                                const size_t width,
                                const size_t height,
                                const size_t depth )
{
    std::cout << "populating convolutional layer " << m_filter_size << " " << m_filter_stride << std::endl;

    std::cout << width << " " << height << " " << prev_layer->width() << " " << prev_layer->height() << std::endl;

    if ( ( width != ( prev_layer->width() - m_filter_size + 1 ) ) ||
        ( height != ( prev_layer->height() - m_filter_size + 1 ) ) )
    {
        std::cerr << "conv_layer_bnu::populate - zero padding not managed for now, \
            so layer size should be consistent with filter size and previous layer size" << std::endl;
        throw network_exception( "inconsistent convolutional layer size" );
    }

    m_prev_layer = prev_layer;

    m_filters.resize( boost::extents[depth][prev_layer->depth()] );
    m_filters_delta.resize( boost::extents[depth][prev_layer->depth()] );
    m_feature_maps.resize( boost::extents[depth] );
    m_error_maps.resize( boost::extents[depth] );

    for ( size_t i = 0; i<depth; i++ )
    {
        m_feature_maps[i] = matrixF( width, height );
        m_error_maps[i] = matrixF( width, height );
        m_error_maps[i].clear();
        for ( auto& _filter : m_filters[i] )
        {
            _filter = matrixF( m_filter_size, m_filter_size );
            random_normal_init( _filter, 1.f / std::sqrt( m_filter_size * m_filter_size ) );
        }
        for ( auto& _filter_delta : m_filters_delta[i] )
        {
            _filter_delta = matrixF( m_filter_size, m_filter_size );
            _filter_delta.clear();
        }
    }
}

void conv_layer_bnu::_convolve_add( const matrixF& prev_feature_map,
                                    const matrixF& filter, const size_t stride,
                                    matrixF& feature_map )
{
    using namespace boost::numeric::ublas;

    // assumption stepsX = stepsY could be easily made...
    auto stepsX = prev_feature_map.size1() - filter.size1() + 1;
    auto stepsY = prev_feature_map.size2() - filter.size2() + 1;
    for ( auto j=0; j<stepsY; j++ )
        for ( auto i=0; i<stepsX; i++ )
        {
            matrixF conv = element_prod( filter,
                project( prev_feature_map,
                    range( i, i+m_filter_size ),
                    range( j, j+m_filter_size ) ) );

            feature_map(i,j) += std::accumulate( conv.data().begin(), conv.data().end(), 0.f );
        }
}

void conv_layer_bnu::feed_forward()
{
    // TODO-CNN : what if previous layer has no feature maps!

    int i = 0;
    for ( auto& feature_map : m_feature_maps )
    {
        feature_map.clear();

        for ( auto j=0; j<m_prev_layer->depth(); i++ )
        {
            const matrixF& prev_feature_map = m_prev_layer->feature_map(j);
            const matrixF& filter = m_filters[i][j];

            // TODO-CNN : wrong convolution, use new convolution class and DO FLIP filter!!!
            _convolve_add( prev_feature_map, filter, m_filter_stride, feature_map );
        }
        i++;
    }
}

void conv_layer_bnu::prepare_training()
{
    // TODO-CNN : : clear errors, deltas...?
}

void conv_layer_bnu::back_propagate()
{
    // Compute errors

    for ( auto j=0; j<m_prev_layer->depth(); j++ )
    {
        const matrixF& prev_feature_map = m_prev_layer->feature_map(j);
        matrixF& prev_error_map = m_prev_layer->error_map(j);

        int i = 0;
        for ( auto& feature_map : m_feature_maps )
        {
            const matrixF& filter = m_filters[i][j];
            // TODO-CNN : wrong convolution, use new convolution class and DON'T FLIP filter!!!
            _convolve_add( m_error_maps[i], filter, m_filter_stride, prev_error_map );
            i++;
        }

        // multiply by sigma derivative
        prev_error_map = bnu::element_prod(
            bnu::element_prod(  prev_feature_map,
                                ( bnu::scalar_matrix<float>(    prev_feature_map.size1(),
                                                                prev_feature_map.size2(), 1.f ) - prev_feature_map ) ),
            prev_error_map
        );
    }

    // Compute gradients

    matrixF grad( m_filter_size, m_filter_size );

    for ( auto i = 0; i < m_filters_delta.shape()[0]; i++ )
    {
        const matrixF& feature_map = m_feature_maps[i];

        for ( auto j = 0; j < m_filters_delta.shape()[1]; j++ )
        {
            const matrixF& prev_error_map = m_prev_layer->error_map(j);

            // TODO-CNN : wrong convolution, use new convolution class and DO FLIP feature_map!!!
            _convolve_add( prev_error_map, feature_map, m_filter_stride, grad );

            m_filters_delta[i][j] += grad / static_cast<float>( m_filters_delta.shape()[1] );
        }
    }
}

void conv_layer_bnu::gradient_descent( boost::shared_ptr<optimizer> optimizer )
{
    // Update weights and bias according to gradients

    // TODO-CNN common to all layers??

    for ( auto i = 0; i < m_filters_delta.shape()[0]; i++ )
    {
        for ( auto j = 0; j < m_filters_delta.shape()[1]; j++ )
        {
            optimizer->update( m_filters[i][j], m_filters_delta[i][j] );
            // TODO-CNN : no bias managed for now
			//m_layers[i].bias() -= m_learning_rate * ( invm * m_layers[i].b_deltas() );*/
        }
    }
}

}; //namespace neurocl
