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
#include "input_layer_bnu.h"
#include "conv_layer_bnu.h"
#include "full_layer_bnu.h"
#include "pool_layer_bnu.h"

// LeNet-5 : http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf

namespace neurocl {

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

    input_layer_bnu m_layer_input;
    conv_layer_bnu m_layer_c1;
    pool_layer_bnu m_layer_s2;
    conv_layer_bnu m_layer_c3;
    pool_layer_bnu m_layer_s4;
    conv_layer_bnu m_layer_c5;
    full_layer_bnu m_layer_f6;
    full_layer_bnu m_layer_output;

    // temporary storage during proof of concept
    std::vector<layer_bnu*> m_layers;

	boost::shared_ptr<optimizer> m_optimizer;
};

} //namespace neurocl

#endif //LENET_BNU_H
