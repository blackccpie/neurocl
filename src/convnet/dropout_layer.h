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

#ifndef DROPOUT_LAYER_H
#define DROPOUT_LAYER_H

#include "layer.h"

#include "common/logger.h"

namespace neurocl { namespace convnet {

using nto = neurocl::convnet::tensor_operation;

class dropout_layer  : public layer
{
public:

    dropout_layer( const std::string& name ) : m_name( name ), m_dropout( 0.5f ) {}
	virtual ~dropout_layer() {}

	virtual const std::string type() const override { return "dropout " + m_name; }

    virtual tensor d_activation( const tensor& in ) const final { return m_prev_layer->d_activation( in ); }

    void populate(  const std::shared_ptr<layer>& prev_layer,
                    const size_t width,
                    const size_t height,
                    const size_t depth )
    {
        LOGGER(info) << "dropout_layer::populate - populating dropout layer " << m_name << std::endl;

        if ( prev_layer->depth() != depth )
            throw network_exception( "invalid depth for dropout layer, should be same depth as previous layer" );

        if ( ( prev_layer->width() != width ) || ( prev_layer->height() != height ) )
            throw network_exception( "invalid size for dropout layer, should be same size as previous layer" );

        m_prev_layer = prev_layer;

        m_feature_maps.resize( width, height, 1, depth );
        m_error_maps.resize( width, height, 1, depth );
        m_mask.resize( width, height, 1, depth );

		// generate initial mask
        nto::bernoulli( m_mask, 1.f - m_dropout );
    }

    virtual size_t width() const override { return m_feature_maps.w(); }
    virtual size_t height() const override { return m_feature_maps.h(); }
    virtual size_t depth() const override { return m_feature_maps.d2(); }

    virtual size_t nb_weights() const override { return 0; }
    virtual size_t nb_bias() const override { return 0; }

    virtual const tensor& feature_maps() const override
        { return m_feature_maps; }

    virtual void feed_forward() override
    {
        if ( m_training )
        	m_feature_maps = ( 1.f / ( 1.f - m_dropout ) ) * nto::elemul( m_mask, m_prev_layer->feature_maps() );
        else
            m_feature_maps = m_prev_layer->feature_maps();
    }

    virtual void back_propagate() override
    {
        m_prev_layer->error_maps({}) = nto::elemul( m_mask, m_error_maps );
    }

    virtual void update_gradients() override
    {
        // NOTHING TO DO : POOL LAYER DOES NOT MANAGE GRADIENTS
    }

	virtual void clear_gradients() override
    {
		// NOTHING TO DO : POOL LAYER DOES NOT MANAGE GRADIENTS
    }

    virtual void gradient_descent( const std::shared_ptr<tensor_solver_iface>& solver ) override
    {
        // pool layer does not manage gradients, but it has to generate new mask after each iteration
        nto::bernoulli( m_mask, 1.f - m_dropout );
    }

    // Fill weights
    virtual void fill_w( const size_t data_size, const float* data ) override { /* NOTHING TO DO */ }
    virtual void fill_w( float* data ) override { /* NOTHING TO DO */ }

    // Fill bias
    virtual void fill_b( const size_t data_size, const float* data ) override { /* NOTHING TO DO */ }
    virtual void fill_b( float* data ) override { /* NOTHING TO DO */ }

    virtual tensor& error_maps( key_errors ) override
        { return m_error_maps; }

protected:

    virtual size_t fan_in() const final
    {
        return m_prev_layer->width() * m_prev_layer->height();
    }

private:

    float m_dropout;

    std::shared_ptr<layer> m_prev_layer;

    tensor m_feature_maps;
    tensor m_error_maps;
    tensor m_mask;

    const std::string m_name;
};

} /*namespace neurocl*/ } /*namespace convnet*/

#endif //POOL_LAYER_BNU_H
