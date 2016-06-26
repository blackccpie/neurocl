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

#ifndef CONV_LAYER_BNU_H
#define CONV_LAYER_BNU_H

#include "layer_bnu.h"

namespace neurocl {

class conv_layer_bnu  : public layer_bnu
{
public:

    conv_layer_bnu();
	virtual ~conv_layer_bnu() {}

    void set_filter_size( const size_t filter_size, const size_t filter_stride = 1 );
    void populate(  layer_bnu* prev_layer,
                    const size_t width,
                    const size_t height,
                    const size_t depth );

    virtual bool has_feature_maps() const override { return true; }

    virtual size_t width() const override { return m_feature_maps[0].size1(); };
    virtual size_t height() const override { return m_feature_maps[0].size2(); };
    virtual size_t depth() const override { return m_feature_maps.shape()[0]; }

    virtual const matrixF& feature_map( const int depth ) const override
        { return m_feature_maps[depth]; }

    virtual void prepare_training() override;
    virtual void feed_forward() override;
    virtual void back_propagate() override;
    virtual void gradient_descent( const boost::shared_ptr<optimizer>& optimizer ) override;

protected:

    virtual matrixF& error_map( const int depth ) override
        { return m_error_maps[depth]; }

private:

    void _convolve_add( const matrixF& prev_feature_map,
                        const matrixF& filter, const size_t stride,
                        matrixF& feature_map );

private:

    layer_bnu* m_prev_layer;

    size_t m_filter_size;
    size_t m_filter_stride;

    marray2F m_filters;
    marray2F m_filters_delta;
    marray1F m_feature_maps;
    marray1F m_error_maps;
};

} //namespace neurocl

#endif //CONV_LAYER_BNU_H
