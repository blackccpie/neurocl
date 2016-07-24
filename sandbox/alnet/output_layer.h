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

#ifndef OUTPUT_LAYER_H
#define OUTPUT_LAYER_H

#include "layer.h"

namespace neurocl {

class output_layer : public layer
{
public:

    output_layer() {}
    virtual ~output_layer() {}

     virtual const std::string type() const override { return "output"; }

    virtual size_t width() const override { return m_feature_maps.w(); }
    virtual size_t height() const override { return m_feature_maps.h(); }
    virtual size_t depth() const override { return m_feature_maps.d2(); }

    void populate(  const std::shared_ptr<layer>& prev_layer,
                    const size_t width,
                    const size_t height,
                    const size_t depth )
    {
        std::cout << "populating output layer " << std::endl;

        m_prev_layer = prev_layer;

        m_feature_maps.resize( width, height, 1, depth );
        m_error_maps.resize( width, height, 1, depth );
    }

    void fill(  const size_t depth1,
                const size_t depth2,
                const size_t data_size,
                const float* data )
    {
        m_feature_maps.fill( depth1, depth2, data_size, data );
    }

    virtual const tensor& feature_maps() const override
        { return m_feature_maps; }

    virtual void prepare_training() override
    {
        // TODO-CNN
    }

    virtual void feed_forward() override
    {
        // TODO-CNN
    }

    virtual void back_propagate() override
    {
        // TODO-CNN

        // Output layer error vector
//        layer_bnu& output_layer = m_layers.back();
//        output_layer.errors() = bnu::element_prod(
//                bnu::element_prod(  output_layer.activations(),
//                                    ( bnu::scalar_vector<float>( output_layer.activations().size(), 1.f ) - output_layer.activations() ) ),
//                ( output_layer.activations() - m_training_output ) );
    }

    virtual void update_gradients() override
    {
        // NOTHING TO DO FOR OUTPUT LAYER
    }

    virtual void gradient_descent( const std::shared_ptr<optimizer>& optimizer ) override
    {
        // NOTHING TO DO FOR OUTPUT LAYER
    }

protected:

    virtual tensor& error_maps() override
        { return m_error_maps; }

private:

    std::shared_ptr<layer> m_prev_layer;

    tensor m_feature_maps;
    tensor m_error_maps;
};

} //namespace neurocl

#endif //OUTPUT_LAYER_H
