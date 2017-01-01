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

#ifndef FULL_LAYER_H
#define FULL_LAYER_H

#include "layer.h"
#include "tensor_tank.h"

#include "common/logger.h"

namespace neurocl { namespace convnet {

using nto = neurocl::convnet::tensor_operation;

class full_layer_iface  : public layer
{
public:

    // populate layer
    virtual void populate(  const std::shared_ptr<layer>& prev_layer,
                            const size_t width,
                            const size_t height,
                            const size_t depth,
                            const size_t cache_size ) = 0;
};

template<class activationT>
class full_layer : public full_layer_iface
{
public:

    full_layer( const std::string& name ) : m_name( name ), m_prev_group_features( false ),
    	m_weights( nullptr ), m_deltas_weights( nullptr ),
    	m_bias( nullptr ), m_deltas_bias( nullptr ) {}

    virtual ~full_layer() {}

    virtual const std::string type() const override { return "full " + m_name; }

    virtual void populate(  const std::shared_ptr<layer>& prev_layer,
                            const size_t width,
                            const size_t height,
                            const size_t depth,
                            const size_t cache_size ) final
    {
        LOGGER(info) << "full_layer::populate - populating full layer " << m_name << std::endl;

        m_prev_layer = prev_layer;

        // check if we have to group previous layer feature maps
        // grouping is only allowed if current depth is 1
        if ( m_prev_layer->depth() != depth )
        {
            if ( depth > 1 )
                throw network_exception( "depth mismatch between full layer and previous layer" );
            else
                m_prev_group_features = true;
        }

        size_t k_group = m_prev_group_features ? m_prev_layer->depth() : 1;
        size_t prev_layer_size = k_group * prev_layer->width() * prev_layer->height();

        m_feature_maps.resize( width, height, 1, depth );
        m_error_maps.resize( width, height, 1, depth );

        if ( m_shared )
        {
            m_bias = tensor_tank::instance().get_shared( "bias", width, height, 1, depth );
            m_bias->fill_random( 1 ); // stddev 1 for bias
            m_deltas_bias = tensor_tank::instance().get_cumulative( "bias_delta", width, height, 1, depth );
            m_bias_cache.resize( cache_size ); int i = 0;
            for ( auto& _bias : m_bias_cache )
                _bias = tensor_tank::instance().get_shared( "bias_cache" + std::to_string(i++), width, height, 1, depth );

            m_weights = tensor_tank::instance().get_shared( "weights", width * height, prev_layer_size, 1, depth );
            m_weights->fill_random( prev_layer_size/*nin*/ );
            m_deltas_weights = tensor_tank::instance().get_cumulative( "weights_delta", width * height, prev_layer_size, 1, depth );
            m_weights_cache.resize( cache_size ); int j = 0;
            for ( auto& _weights : m_weights_cache )
                _weights = tensor_tank::instance().get_shared( "weights_cache" + std::to_string(j++), width * height, prev_layer_size, 1, depth );
        }
        else
        {
        	m_bias = tensor_tank::instance().get_standard( "bias", width, height, 1, depth );
            m_bias->fill_random( 1 ); // stddev 1 for bias
        	m_deltas_bias = tensor_tank::instance().get_standard( "bias_delta", width, height, 1, depth );
            m_bias_cache.resize( cache_size ); int i = 0;
            for ( auto& _bias : m_bias_cache )
                _bias = tensor_tank::instance().get_standard( "bias_cache" + std::to_string(i++), width, height, 1, depth );

        	m_weights = tensor_tank::instance().get_standard( "weights", width * height, prev_layer_size, 1, depth );
            m_weights->fill_random( prev_layer_size/*nin*/ );
        	m_deltas_weights = tensor_tank::instance().get_standard( "weights_delta", width * height, prev_layer_size, 1, depth );
            m_weights_cache.resize( cache_size ); int j = 0;
            for ( auto& _weights : m_weights_cache )
                _weights = tensor_tank::instance().get_standard( "weights_cache" + std::to_string(j++), width * height, prev_layer_size, 1, depth );
        }
    }

    virtual size_t width() const override { return m_feature_maps.w(); }
    virtual size_t height() const override { return m_feature_maps.h(); }
    virtual size_t depth() const override { return m_feature_maps.d2(); }

    virtual size_t nb_weights() const override { return m_weights->w() * m_weights->h() * m_weights->d1() * m_weights->d2(); }
    virtual size_t nb_bias() const override { return m_bias->w() * m_bias->h() * m_bias->d1() * m_bias->d2(); }

    virtual const tensor& feature_maps() const override
        { return m_feature_maps; }

    virtual void feed_forward() override
    {
        const auto& prev_feature_maps = m_prev_layer->feature_maps();

        if ( m_prev_group_features )
        {
            const tensor grouped_feature_maps = nto::group( prev_feature_maps );

            // apply weights and bias
            m_feature_maps = nto::muladd( *m_weights, grouped_feature_maps, *m_bias );
        }
        else
        {
        	// apply weights and bias
        	m_feature_maps = nto::muladd( *m_weights, prev_feature_maps, *m_bias );
        }

        // apply activation function
        activationT::f( m_feature_maps );
    }

    virtual void back_propagate() override
    {
        const tensor& prev_feature_maps = m_prev_layer->feature_maps();
        tensor& prev_error_maps = m_prev_layer->error_maps({});

        // Need to back prop?
        if ( prev_error_maps.empty() )
            return;

        // Compute errors

        if ( m_prev_group_features )
        {
            const auto&& grouped_feature_maps = nto::group( prev_feature_maps );

            const tensor grouped_error_maps = nto::elemul(
                activationT::d_f( grouped_feature_maps ),
                nto::multrans1( *m_weights, m_error_maps )
            );

            nto::ungroup( grouped_error_maps, prev_error_maps );
        }
        else
        {
        	prev_error_maps = nto::elemul(
            	activationT::d_f( prev_feature_maps ),
            	nto::multrans1( *m_weights, m_error_maps )
        	);
        }
    }

    virtual void update_gradients() override
    {
        // Compute gradients

        if ( m_prev_group_features )
        {
            const auto&& grouped_feature_maps = nto::group( m_prev_layer->feature_maps() );

            *m_deltas_weights += nto::multrans2( m_error_maps, grouped_feature_maps );
        	*m_deltas_bias += m_error_maps;
        }
        else
        {
            *m_deltas_weights += nto::multrans2( m_error_maps, m_prev_layer->feature_maps() );
            *m_deltas_bias += m_error_maps;
        }
    }

    virtual void clear_gradients() override
    {
        m_deltas_weights->clear();
        m_deltas_bias->clear();
    }

    virtual void gradient_descent( const std::shared_ptr<tensor_solver_iface>& solver ) override
    {
        // Optimize gradients

        nto::optimize<nto::optimize_mode::std>( solver, m_weights, m_weights_cache.data(), m_deltas_weights );
        nto::optimize<nto::optimize_mode::redux>( solver, m_bias, m_bias_cache.data(), m_deltas_bias );
    }

    // Fill weights
    virtual void fill_w( const size_t data_size, const float* data ) override
    {
         m_weights->grouped_fill( data_size, data );
    }
    virtual void fill_w( float* data ) override
    {
         m_weights->grouped_fill( data );
    }

    // Fill bias
    virtual void fill_b( const size_t data_size, const float* data ) override
    {
         m_bias->grouped_fill( data_size, data );
    }
    virtual void fill_b( float* data ) override
    {
         m_bias->grouped_fill( data );
    }

    //! get gradient checker
    virtual std::unique_ptr<tensor_gradient_checker> get_gradient_checker() final
    {
        return std::move( std::unique_ptr<tensor_gradient_checker>(
            new tensor_gradient_checker( *m_weights, *m_deltas_weights ) ) );
    }

protected:

    virtual tensor& error_maps( key_errors ) override
        { return m_error_maps; }

private:

    std::shared_ptr<layer> m_prev_layer;

    tensor m_feature_maps;
    tensor m_error_maps;

    tensor* m_bias;
    tensor* m_deltas_bias;
    std::vector<tensor*> m_bias_cache;

    tensor* m_weights;
    tensor* m_deltas_weights;
    std::vector<tensor*> m_weights_cache;

    bool m_prev_group_features;

    const std::string m_name;
};

} /*namespace neurocl*/ } /*namespace convnet*/

#endif //FULL_LAYER_BNU_H
