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

namespace neurocl {

using nto = neurocl::tensor_operation;

class conv_layer  : public layer
{
public:

    conv_layer() {}
	virtual ~conv_layer() {}

    void set_filter_size( const size_t filter_size, const size_t filter_stride = 1 );
    void populate(  const std::shared_ptr<layer>& prev_layer,
                    const size_t width,
                    const size_t height,
                    const size_t depth )
    {
        m_prev_layer = prev_layer;
    }

    virtual size_t width() const override { return 0; };
    virtual size_t height() const override { return 0; };
    virtual size_t depth() const override { return 0; }

    virtual const tensor& feature_maps() const override
        { return m_feature_maps; }

    virtual void prepare_training() override
    {

    }
    virtual void feed_forward() override
    {
        nto::convolve_add<nto::kernel_flip,nto::pad_valid>(
            m_prev_layer->feature_maps(),
            m_filters,
            m_feature_maps,
            m_filter_stride );

        nto::relu( m_feature_maps );
    }
    virtual void back_propagate() override
    {
        // Compute errors

        nto::convolve_add<nto::kernel_std,nto::pad_full>( //padding?????
            m_error_maps,
            m_filters,
            m_prev_layer->error_maps(),
            m_filter_stride );

        nto::d_relu( m_prev_layer->error_maps(), m_prev_layer->feature_maps() );

        // Compute gradients

        tensor grad;
        nto::convolve_add<nto::kernel_flip,nto::pad_full>( //padding?????
            m_prev_layer->error_maps(),
            m_prev_layer->feature_maps(),
            grad,
            m_filter_stride);

        //m_filters_delta += grad / static_cast<float>( m_filters_delta.shape()[1] );
    }
    virtual void gradient_descent( const std::shared_ptr<optimizer>& optimizer ) override
    {
        nto::optimize( optimizer, m_filters, m_filters_delta );
    }

protected:

    virtual tensor& error_maps() override
        { return m_error_maps; }

private:

    std::shared_ptr<layer> m_prev_layer;

    size_t m_filter_size;
    size_t m_filter_stride;

    tensor m_filters;
    tensor m_filters_delta;
    tensor m_feature_maps;
    tensor m_error_maps;
};

} //namespace neurocl

#endif //CONV_LAYER_H
