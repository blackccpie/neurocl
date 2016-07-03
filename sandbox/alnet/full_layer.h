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

namespace neurocl {

class full_layer : public layer
{
public:

    full_layer() {}
	virtual ~full_layer() {}

    void populate(  const std::shared_ptr<layer>& prev_layer,
                    const size_t width,
                    const size_t height,
                    const size_t depth )
    {

    }

    virtual size_t width() const override { return 0; };
    virtual size_t height() const override { return 0; }
    virtual size_t depth() const override { return 0; }

    virtual const tensor& feature_maps() const override
        { return m_feature_maps; }

    virtual void prepare_training() override
    {

    }

    virtual void feed_forward() override
    {
        const tensor& prev_feature_maps = m_prev_layer->feature_maps();

        // apply weights and bias
        m_feature_maps = nto::muladd( m_weights, prev_feature_maps, m_bias );

        // apply sigmoid function
        nto::relu( m_feature_maps );
    }

    virtual void back_propagate() override
    {
        // Compute errors

        // m_prev_layer->error_map(0) = bnu::element_prod(
        //                                     bnu::element_prod(
        //                                         m_feature_map,
        //                                         ( bnu::scalar_matrix<float>( m_feature_map.size1() * m_feature_map.size2(), 1.f ) - m_feature_map ) ),
        //                                     bnu::prod(
        //                                         bnu::trans( m_weights ),
        //                                         m_error_map ) );

        // Compute gradients

        /*
        m_layers[i].w_deltas() = m_layers[i].w_deltas() + bnu::outer_prod( m_layers[i+1].errors(), m_layers[i].activations() );
        m_layers[i].b_deltas() = m_layers[i].b_deltas() + m_layers[i+1].errors();
        */
    }

    virtual void gradient_descent( const std::shared_ptr<optimizer>& optimizer ) override
    {

    }

protected:

    virtual tensor& error_maps() override
        { return m_error_maps; }

private:

    std::shared_ptr<layer> m_prev_layer;

    tensor m_feature_maps;
    tensor m_error_maps;
    tensor m_bias;
    tensor m_deltas_bias;

    tensor m_weights;
    tensor m_deltas_weights;
};

} //namespace neurocl

#endif //FULL_LAYER_BNU_H
