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

#ifndef LENET_BNU_H
#define LENET_BNU_H

#include "network_interface.h"

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>

typedef typename boost::numeric::ublas::vector<float> vectorF;
typedef typename boost::numeric::ublas::matrix<float> matrixF;

namespace neurocl {

class layer_iface
{
public:

    virtual void feed_forward( const layer_iface* prev_layer ) = 0;
};

class full_layer_bnu : public layer_iface
{
public:

    full_layer_bnu();
	virtual ~full_layer_bnu() {}

    void populate( const layer_size& cur_layer_size, const layer_size& next_layer_size );

    virtual void feed_forward( const layer_iface* prev_layer );

    vectorF& bias() { return m_bias; }
    vectorF& activations() { return m_activations; }
    matrixF& weights() { return m_output_weights; }
    vectorF& errors() { return m_errors; }
    matrixF& w_deltas() { return m_deltas_weight; }
    vectorF& b_deltas() { return m_deltas_bias; }

    const std::string dump_weights() const;
    const std::string dump_bias() const;
    const std::string dump_activations() const;

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

class conv_layer_bnu  : public layer_iface
{
public:

    conv_layer_bnu();
	virtual ~conv_layer_bnu() {}

    void set_filter_size( const size_t filter_size, const size_t filter_stride = 1 );
    void populate( const size_t width, const size_t height, const size_t depth );

    virtual void feed_forward( const layer_iface* prev_layer );

private:

    size_t m_filter_size;
    size_t m_filter_stride;

    boost::shared_array<matrixF> m_filters;
    boost::shared_array<matrixF> m_feature_maps;
};

class pool_layer_bnu  : public layer_iface
{
public:

    pool_layer_bnu();
	virtual ~pool_layer_bnu() {}

    void populate( const size_t width, const size_t height, const size_t depth );

    virtual void feed_forward( const layer_iface* prev_layer );

private:

    boost::shared_array<matrixF> m_feature_maps;
};

class lenet_bnu final : public network_interface
{
public:

	lenet_bnu();
	virtual ~lenet_bnu() {}

    void add_layers_2d( const std::vector<layer_size>& layer_sizes );

    void set_input(  const size_t& in_size, const float* in );
    void set_output( const size_t& out_size, const float* out );

    void prepare_training();

    // pure compute-critic virtuals to be implemented in inherited classes
    void feed_forward();
    void back_propagate();
    void gradient_descent();

    const size_t count_layers()
    {
        /* STUBBED FOR NOW*/
        return 8; /*return m_layers.size();*/
    }
    const layer_ptr get_layer_ptr( const size_t layer_idx );
    void set_layer_ptr( const size_t layer_idx, const layer_ptr& layer );

    const output_ptr output();

    const std::string dump_weights();
    const std::string dump_bias();
    const std::string dump_activations();

protected:

    size_t m_training_samples;

    vectorF m_training_output;

    full_layer_bnu m_layer_input;
    conv_layer_bnu m_layer_c1;
    pool_layer_bnu m_layer_s2;
    conv_layer_bnu m_layer_c3;
    pool_layer_bnu m_layer_s4;
    full_layer_bnu m_layer_c5;
    full_layer_bnu m_layer_f6;
    full_layer_bnu m_layer_output;

    // temporary storage during proof of concept
    std::vector<layer_iface*> m_layers;

    float m_learning_rate;  // [0.0..1.0]
    float m_weight_decay;   // [0.0..1.0]
};

} //namespace neurocl

#endif //LENET_BNU_H
