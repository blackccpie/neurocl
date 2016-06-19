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

#include "pool_layer_bnu.h"
#include "network_exception.h"

namespace neurocl {

pool_layer_bnu::pool_layer_bnu() : m_subsample( 1 )
{
}

void pool_layer_bnu::populate(  const layer_bnu* prev_layer,
                                const size_t width,
                                const size_t height,
                                const size_t depth )
{
    std::cout << "populating pooling layer" << std::endl;

    m_prev_layer = prev_layer;

    // compute subsampling rate, throw error if not integer
    if ( ( prev_layer->width() % width) == 0 )
        m_subsample = prev_layer->width() / width;
    else
        throw network_exception( "invalid subsampling for max pooling" );

    m_feature_maps.resize( boost::extents[depth] );
    for ( auto& _feature : m_feature_maps )
    {
        _feature = matrixF( width, height );
    }
}

void pool_layer_bnu::feed_forward()
{
    for ( auto i = 0; i < m_feature_maps.shape()[0]; i++ )
    {
        const matrixF& prev_feature_map = m_prev_layer->feature_map(i);
        matrixF& feature_map = m_feature_maps[i];
        auto prev_width = prev_feature_map.size1();
        auto prev_it1 = prev_feature_map.begin1();
        for( auto it1 = feature_map.begin1(); it1 != feature_map.end1(); it1++, prev_it1 += m_subsample )
        {
            auto prev_it2 = prev_it1.begin();
            for( auto it2 = it1.begin(); it2 !=it1.end(); it2++, prev_it2 += m_subsample )
            {
                float max_value = std::numeric_limits<float_t>::lowest();

                // could use ublas::project + std::accumulate + std::max for more compact expression

                // compute max in subsampling zone
                for ( auto i =0; i<m_subsample; i++ )
                    for ( auto j =0; j<m_subsample; j++ )
                    {
                        const float& value = *(prev_it2 + i + (j*prev_width) );
                        if ( value > max_value )
                            max_value = value;
                    }

                // update value in the destination feature map
                *it2 = max_value;
            }
        }
    }
}

void pool_layer_bnu::back_propagate()
{
    // TODO-CNN : what if previous layer has no error maps!

    for ( auto i = 0; i < m_feature_maps.shape()[0]; i++ )
    {
        const matrixF& prev_feature_map = m_prev_layer->feature_map(i);

        matrixF& prev_error_map = m_prev_layer->error_map(i);

        const matrixF& error_map = m_error_maps[i];

		// NOT IMPLEMEMENTED YET
    }
}

}; //namespace neurocl
