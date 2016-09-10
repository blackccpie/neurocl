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

#ifndef POOL_LAYER_H
#define POOL_LAYER_H

#include "layer.h"

#include "common/network_exception.h"

namespace neurocl { namespace convnet {

class pool_layer  : public layer
{
public:

    pool_layer( const std::string& name ) : m_name( name ) {}
	virtual ~pool_layer() {}

	virtual const std::string type() const override { return "pool " + m_name; }

    void populate(  const std::shared_ptr<layer>& prev_layer,
                    const size_t width,
                    const size_t height,
                    const size_t depth )
    {
        std::cout << "populating pooling layer" << std::endl;

        // compute subsampling rate, throw error if not integer
        if ( ( prev_layer->width() % width) == 0 )
            m_subsample = prev_layer->width() / width;
        else
            throw network_exception( "invalid subsampling for max pooling" );

        m_prev_layer = prev_layer;

        m_feature_maps.resize( width, height, 1, depth );
        m_error_maps.resize( width, height, 1, depth );
    }

    virtual size_t width() const override { return m_feature_maps.w(); }
    virtual size_t height() const override { return m_feature_maps.h(); }
    virtual size_t depth() const override { return m_feature_maps.d2(); }

    virtual size_t nb_weights() const override { return 0; }
    virtual size_t nb_bias() const override { return 0; }

    virtual const tensor& feature_maps() const override
        { return m_feature_maps; }

    virtual void prepare_training() override
    {

    }

    virtual void feed_forward() override
    {
        m_feature_maps = nto::subsample( m_prev_layer->feature_maps(), m_subsample );
    }

    virtual void back_propagate() override
    {
        m_prev_layer->error_maps() = nto::d_subsample( m_error_maps, m_prev_layer->feature_maps(), m_subsample );
    }

    virtual void update_gradients() override
    {
        // NOTHING TO DO : POOL LAYER DOES NOT MANAGE GRADIENTS
    }

    virtual void gradient_descent( const std::shared_ptr<optimizer>& optimizer ) override
    {
        // NOTHING TO DO : POOL LAYER DOES NOT MANAGE GRADIENTS
    }

    // Fill weights
    virtual void fill_w( const size_t data_size, const float* data ) override { /* NOTHING TO DO */ }
    virtual void fill_w( float* data ) override { /* NOTHING TO DO */ }

    // Fill bias
    virtual void fill_b( const size_t data_size, const float* data ) override { /* NOTHING TO DO */ }
    virtual void fill_b( float* data ) override { /* NOTHING TO DO */ }

protected:

    virtual tensor& error_maps() override
        { return m_error_maps; }

private:

    size_t m_subsample;

    std::shared_ptr<layer> m_prev_layer;

    tensor m_feature_maps;
    tensor m_error_maps;

    const std::string m_name;
};

} /*namespace neurocl*/ } /*namespace convnet*/

#endif //POOL_LAYER_BNU_H
