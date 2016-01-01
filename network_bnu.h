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

#ifndef NETWORK_BNU_H
#define NETWORK_BNU_H

#include "network_interface.h"

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>

typedef typename boost::numeric::ublas::vector<float> vectorF;
typedef typename boost::numeric::ublas::matrix<float> matrixF;

namespace neurocl {

class layer_bnu
{
public:

    layer_bnu();
	virtual ~layer_bnu() {}

    void populate( const layer_size& cur_layer_size, const layer_size& next_layer_size );

    vectorF& bias() { return m_bias; }
    vectorF& activations() { return m_activations; }
    matrixF& weights() { return m_output_weights; }
    vectorF& errors() { return m_errors; }
    matrixF& w_deltas() { return m_deltas_weight; }
    vectorF& b_deltas() { return m_deltas_bias; }

private:

    vectorF m_activations;
    vectorF m_errors;
    vectorF m_bias;
    vectorF m_deltas_bias;

    // We follow stanford convention:
    // http://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf
    matrixF m_output_weights;
    matrixF m_deltas_weight;
};

class network_bnu : public network_interface
{
public:

	network_bnu();
	virtual ~network_bnu() {}

    // Convention : input layer is index 0
    void add_layers_2d( const std::vector<layer_size>& layer_sizes );

    void set_input_sample(  const size_t& isample_size, const float* isample,
                            const size_t& osample_size, const float* osample );

    void feed_forward();
    void gradient_descent();

    const float output();

private:

    void _back_propagate();
    void _gradient_descent();

private:

    float m_learning_rate;  // [0.0..1.0]
    float m_weight_decay;   // [0.0..1.0]

    vectorF m_training_output;

    std::vector<layer_bnu> m_layers;
};

} //namespace neurocl

#endif //NETWORK_BNU_H
