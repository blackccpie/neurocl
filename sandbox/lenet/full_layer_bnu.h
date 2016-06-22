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

#ifndef FULL_LAYER_BNU_H
#define FULL_LAYER_BNU_H

#include "layer_bnu.h"

namespace neurocl {

class full_layer_bnu : public layer_bnu
{
public:

    full_layer_bnu();
	virtual ~full_layer_bnu() {}

    void populate(  layer_bnu* prev_layer,
                    const layer_size& lsize );

    virtual bool has_feature_maps() const override { return false; }

    virtual size_t width() const override { return m_activations.size(); };
    virtual size_t height() const override { return 1; };
    virtual size_t depth() const override { return 1; }

    virtual const vectorF& activations() const override
        { return m_activations; }
    virtual const matrixF& feature_map( const int depth ) const override
        { return empty::matrix; }

    virtual void prepare_training() override;
    virtual void feed_forward() override;
    virtual void back_propagate() override;
    virtual void gradient_descent() override;

    /*vectorF& bias() { return m_bias; }
    vectorF& activations() { return m_activations; }
    matrixF& weights() { return m_output_weights; }
    vectorF& errors() { return m_errors; }
    matrixF& w_deltas() { return m_deltas_weight; }
    vectorF& b_deltas() { return m_deltas_bias; }*/

private:

    layer_bnu* m_prev_layer;

    vectorF m_activations;
    vectorF m_errors;
    vectorF m_bias;
    vectorF m_deltas_bias;

    // We follow stanford convention:
    // http://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf
    matrixF m_weights;
    matrixF m_deltas_weight;
};

} //namespace neurocl

#endif //FULL_LAYER_BNU_H
