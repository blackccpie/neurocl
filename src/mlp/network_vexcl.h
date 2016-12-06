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

#ifndef NETWORK_VEXCL_H
#define NETWORK_VEXCL_H

#include "network_interface_mlp.h"

#include <vexcl/vexcl.hpp>

namespace neurocl { namespace mlp {

class layer_vexcl
{
public:

    layer_vexcl();
	virtual ~layer_vexcl() {}

    void populate( const layer_size& cur_layer_size, const layer_size& next_layer_size );

    vex::vector<float>& bias() { return m_bias; }
    vex::vector<float>& activations() { return m_activations; }
    vex::vector<float>& weights() { return m_output_weights; }
    vex::vector<float>& errors() { return m_errors; }
    vex::vector<float>& w_deltas() { return m_deltas_weight; }
    vex::vector<float>& b_deltas() { return m_deltas_bias; }

    std::pair<size_t,size_t>& w_size() { return m_weights_size; }

    const std::string dump_weights() const;
    const std::string dump_bias() const;
    const std::string dump_activations() const;

private:

    std::pair<size_t,size_t> m_weights_size;

    vex::vector<float> m_activations;
    vex::vector<float> m_errors;
    vex::vector<float> m_bias;
    vex::vector<float> m_deltas_bias;

    // We follow stanford convention:
    // http://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf
    vex::vector<float> m_output_weights;
    vex::vector<float> m_deltas_weight;
};

class network_vexcl : public network_interface_mlp
{
public:

	network_vexcl();
	virtual ~network_vexcl() {}

    // Convention : input layer is index 0
    virtual void add_layers_2d( const std::vector<layer_size>& layer_sizes ) final;

    virtual void set_input(  const size_t& in_size, const float* in ) final;
    virtual void set_output( const size_t& out_size, const float* out ) final;

	virtual void clear_gradients() final;
    virtual void gradient_check() final;

    virtual void feed_forward() final;
    virtual void back_propagate() final;
    virtual void gradient_descent() final;

    virtual const size_t count_layers() final { return m_layers.size(); }
    virtual const layer_ptr get_layer_ptr( const size_t layer_idx ) final;
    virtual void set_layer_ptr( const size_t layer_idx, const layer_ptr& layer ) final;

    virtual const output_ptr output() final;

    virtual const std::string dump_weights() final;
    virtual const std::string dump_bias() final;
    virtual const std::string dump_activations() final;

private:

    size_t m_training_samples;

    float m_learning_rate;  // [0.0..1.0]
    float m_weight_decay;   // [0.0..1.0]

    vex::vector<float> m_training_output;

    std::vector<layer_vexcl> m_layers;
};

} /*namespace neurocl*/ } /*namespace mlp*/

#endif //NETWORK_VEXCL_H
