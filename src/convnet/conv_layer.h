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

#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include "layer.h"
#include "tensor_tank.h"

#include "common/logger.h"

namespace neurocl { namespace convnet {

using nto = neurocl::convnet::tensor_operation;

class conv_layer_iface  : public layer
{
public:

    // populate layer
    virtual void populate(  const std::shared_ptr<layer>& prev_layer,
                            const size_t width,
                            const size_t height,
                            const size_t depth,
                            const size_t cache_size ) = 0;

    // set filter kernel size
    virtual void set_filter_size( const size_t filter_size, const size_t filter_stride = 1 ) = 0;
};

template<class activationT>
class conv_layer  : public conv_layer_iface
{
public:

    conv_layer( const std::string& name ) : m_name( name ),
    	m_filters( nullptr ), m_deltas_filters( nullptr ),
    	m_bias( nullptr ), m_deltas_bias( nullptr ) {}

    virtual ~conv_layer() {}

    virtual const std::string type() const final { return "conv " + m_name; }

    virtual tensor d_activation( const tensor& in ) const final { return activationT::d_f( in ); }

    virtual void set_filter_size( const size_t filter_size, const size_t filter_stride = 1 ) final
    {
        m_filter_size = filter_size;
        m_filter_stride = filter_stride;
    }

    virtual void populate(  const std::shared_ptr<layer>& prev_layer,
                            const size_t width,
                            const size_t height,
                            const size_t depth,
                            const size_t cache_size) final
    {
        LOGGER(info) << "conv_layer::populate - populating convolutional layer " << m_name << std::endl;

        if ( ( width != ( prev_layer->width() - m_filter_size + 1 ) ) ||
            ( height != ( prev_layer->height() - m_filter_size + 1 ) ) )
        {
            LOGGER(error) << "conv_layer::populate - zero padding not managed for now, "
                "so layer size should be consistent with filter size and previous layer size" << std::endl;
            throw network_exception( "inconsistent convolutional layer size" );
        }

        m_prev_layer = prev_layer;

        m_feature_maps.resize( width, height, 1, depth );
        m_error_maps.resize( width, height, 1, depth );

        if ( m_shared )
        {
            m_bias = tensor_tank::instance().get_shared( "bias", width, height, 1, depth );
            m_bias->uniform_fill_random( 1.f /*stddev*/ ); // uniform because of parameters sharing
            m_deltas_bias = tensor_tank::instance().get_cumulative( "bias_delta", width, height, 1, depth );
            m_bias_cache.resize( cache_size ); int i = 0;
            for ( auto& _bias : m_bias_cache )
                _bias = tensor_tank::instance().get_shared( "bias_cache" + std::to_string(i++), width, height, 1, depth );

        	m_filters = tensor_tank::instance().get_shared( "filters", m_filter_size, m_filter_size, prev_layer->depth(), depth );
            m_filters->fill_random( fan_in() );
        	m_deltas_filters = tensor_tank::instance().get_cumulative( "filters_delta", m_filter_size, m_filter_size, prev_layer->depth(), depth );
            m_filters_cache.resize( cache_size ); int j = 0;
            for ( auto& _filters : m_filters_cache )
                _filters = tensor_tank::instance().get_shared( "filters_cache" + std::to_string(j++), m_filter_size, m_filter_size, prev_layer->depth(), depth );
        }
        else
        {
        	m_bias = tensor_tank::instance().get_standard( "bias", width, height, 1, depth );
        	m_bias->uniform_fill_random( 1.f /*stddev*/ ); // uniform because of parameters sharing
        	m_deltas_bias = tensor_tank::instance().get_standard( "bias_delta", width, height, 1, depth );
            m_bias_cache.resize( cache_size ); int i = 0;
            for ( auto& _bias : m_bias_cache )
        		_bias = tensor_tank::instance().get_standard( "bias_cache" + std::to_string(i++), width, height, 1, depth );

            m_filters = tensor_tank::instance().get_standard( "filters", m_filter_size, m_filter_size, prev_layer->depth(), depth );
            m_filters->fill_random( fan_in() );
            m_deltas_filters = tensor_tank::instance().get_standard( "filters_delta", m_filter_size, m_filter_size, prev_layer->depth(), depth );
            m_filters_cache.resize( cache_size ); int j = 0;
            for ( auto& _filters : m_filters_cache )
                _filters = tensor_tank::instance().get_standard( "filters_cache" + std::to_string(j++), m_filter_size, m_filter_size, prev_layer->depth(), depth );
        }
    }

    virtual size_t width() const override { return m_feature_maps.w(); }
    virtual size_t height() const override { return m_feature_maps.h(); }
    virtual size_t depth() const override { return m_feature_maps.d2(); }

    virtual size_t nb_weights() const override { return m_filters->w() * m_filters->h() * m_filters->d1() * m_filters->d2(); }
    virtual size_t nb_bias() const override { return m_bias->w() * m_bias->h() * m_bias->d1() * m_bias->d2(); }

    virtual const tensor& feature_maps() const override
        { return m_feature_maps; }

    virtual void feed_forward() override
    {
        m_feature_maps = nto::convolve_add_forward<nto::kernel_mode::flip,nto::pad_mode::valid>(
            m_prev_layer->feature_maps(),
            *m_filters,
        	m_filter_stride ) + *m_bias;

		// could be computed in next pooling layer if present for reduced computation
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

        prev_error_maps = nto::convolve_add_backward<nto::kernel_mode::std,nto::pad_mode::full>(
            m_error_maps,
            *m_filters,
            m_filter_stride );

        // multiply by sigma derivative
        prev_error_maps = nto::elemul(
            m_prev_layer->d_activation( prev_feature_maps ),
            prev_error_maps
        );
    }

    virtual void update_gradients() override
    {
        // Compute gradients

        auto&& grad = nto::convolve_update<nto::kernel_mode::flip,nto::pad_mode::valid>(
            m_prev_layer->feature_maps(),
            m_error_maps,
            m_filter_stride);

        *m_deltas_filters += grad / static_cast<float>( m_deltas_filters->d2() );
        *m_deltas_bias += nto::uniform_sum( m_error_maps );
    }

	virtual void clear_gradients() override
    {
        m_deltas_filters->clear();
        m_deltas_bias->clear();
    }

    virtual void gradient_descent( const std::shared_ptr<tensor_solver_iface>& solver ) override
    {
        // Optimize gradients

        nto::optimize<nto::optimize_mode::std>( solver, m_filters, m_filters_cache.data(), m_deltas_filters );
        nto::optimize<nto::optimize_mode::std>( solver, m_bias, m_bias_cache.data(), m_deltas_bias );
    }

    // Fill weights
    virtual void fill_w( const size_t data_size, const float* data ) override
    {
         m_filters->grouped_fill( data_size, data );
    }
    virtual void fill_w( float* data ) override
    {
         m_filters->grouped_fill( data );
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

    virtual tensor& error_maps( key_errors ) override
        { return m_error_maps; }

    //! get gradient checker
    virtual std::unique_ptr<tensor_gradient_checker> get_gradient_checker() final
    {
        return std::move( std::unique_ptr<tensor_gradient_checker>(
            new tensor_gradient_checker( *m_filters, *m_deltas_filters ) ) );
    }

	// copy accessor, not made for performance but rather for network introspection
    virtual tensor weights( key_weights ) final { return *m_filters; }

protected:

    virtual size_t fan_in() const final
    {
        return m_filter_size * m_filter_size;
    }

private:

    std::shared_ptr<layer> m_prev_layer;

    size_t m_filter_size;
    size_t m_filter_stride;

    tensor* m_filters;
    tensor* m_deltas_filters;
    std::vector<tensor*> m_filters_cache;

    tensor* m_bias;
    tensor* m_deltas_bias;
    std::vector<tensor*> m_bias_cache;

    tensor m_feature_maps;
    tensor m_error_maps;

    const std::string m_name;
};

} /*namespace neurocl*/ } /*namespace convnet*/

#endif //CONV_LAYER_H
