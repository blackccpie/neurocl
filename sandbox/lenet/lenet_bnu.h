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

#include <boost/multi_array.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>

typedef typename boost::numeric::ublas::vector<float> vectorF;
typedef typename boost::numeric::ublas::matrix<float> matrixF;

typedef typename boost::multi_array<matrixF,1> marray1F;
typedef typename boost::multi_array<matrixF,2> marray2F;

namespace neurocl {

class layer_iface
{

public:

    //virtual bool is_input() { return false; }

    virtual bool has_feature_maps() const = 0;

    size_t size() const { return width()*height()*depth(); }
    virtual size_t width() const = 0;
    virtual size_t height() const = 0;
    virtual size_t depth() const = 0;

    virtual const vectorF& activations() const = 0;
    virtual const matrixF& feature_map( const int depth ) const = 0;

    virtual void feed_forward() = 0;

protected:

    struct empty
    {
        static const matrixF matrix;
        static const vectorF vector;
    };
};

const matrixF layer_iface::empty::matrix = matrixF();
const vectorF layer_iface::empty::vector = vectorF();

class full_layer_bnu : public layer_iface
{
public:

    full_layer_bnu();
	virtual ~full_layer_bnu() {}

    void populate(  const layer_iface* prev_layer,
                    const layer_size& lsize );

    virtual bool has_feature_maps() const { return false; }

    virtual size_t width() const { return 1; };
    virtual size_t height() const { return 1; };
    virtual size_t depth() const { return 1; }

    virtual const vectorF& activations() const
        { return m_activations; }
    virtual const matrixF& feature_map( const int depth ) const
        { return empty::matrix; }

    virtual void feed_forward();

    /*vectorF& bias() { return m_bias; }
    vectorF& activations() { return m_activations; }
    matrixF& weights() { return m_output_weights; }
    vectorF& errors() { return m_errors; }
    matrixF& w_deltas() { return m_deltas_weight; }
    vectorF& b_deltas() { return m_deltas_bias; }*/

private:

    const layer_iface* m_prev_layer;

    vectorF m_activations;
    vectorF m_errors;
    vectorF m_bias;
    vectorF m_deltas_bias;

    // We follow stanford convention:
    // http://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf
    matrixF m_weights;
    matrixF m_deltas_weight;
};

class conv_layer_bnu  : public layer_iface
{
public:

    conv_layer_bnu();
	virtual ~conv_layer_bnu() {}

    void set_filter_size( const size_t filter_size, const size_t filter_stride = 1 );
    void populate(  const layer_iface* prev_layer,
                    const size_t width,
                    const size_t height,
                    const size_t depth );

    virtual bool has_feature_maps() const { return true; }

    virtual size_t width() const { return m_feature_maps[0].size1(); };
    virtual size_t height() const { return m_feature_maps[0].size2(); };
    virtual size_t depth() const { return m_feature_maps.shape()[0]; }

    virtual const vectorF& activations() const
        { return empty::vector; }
    virtual const matrixF& feature_map( const int depth ) const
        { return m_feature_maps[depth]; }

    virtual void feed_forward();

private:

    void _convolve_add( const matrixF& prev_feature_map,
                        const matrixF& filter, const size_t stride,
                        matrixF& feature_map );

private:

    const layer_iface* m_prev_layer;

    size_t m_filter_size;
    size_t m_filter_stride;

    marray2F m_filters;
    marray1F m_feature_maps;
};

class pool_layer_bnu  : public layer_iface
{
public:

    pool_layer_bnu();
	virtual ~pool_layer_bnu() {}

    void populate(  const layer_iface* prev_layer,
                    const size_t width,
                    const size_t height,
                    const size_t depth );

    virtual bool has_feature_maps() const { return true; }

    virtual size_t width() const { return m_feature_maps[0].size1(); };
    virtual size_t height() const { return m_feature_maps[0].size2(); };
    virtual size_t depth() const { return m_feature_maps.shape()[0]; }

    virtual const vectorF& activations() const
        { return empty::vector; }
    virtual const matrixF& feature_map( const int depth ) const
        { return m_feature_maps[depth]; }

    virtual void feed_forward();

private:

    size_t m_subsample;

    const layer_iface* m_prev_layer;

    marray1F m_feature_maps;
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
