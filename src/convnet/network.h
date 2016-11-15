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

#ifndef NETWORK_CONVNET_H
#define NETWORK_CONVNET_H

#include "network_interface.h"

#include <memory>
#include <vector>

namespace neurocl { namespace convnet {

class layer;
class tensor_solver_iface;

class network : public network_interface
{
public:

	network();
	virtual ~network() {}

    void add_layers( const std::vector<layer_descr>& layers ) override;

	void set_input(  const size_t& in_size, const float* in ) override;
    void set_output( const size_t& out_size, const float* out ) override;
    const output_ptr output() override;

    void feed_forward() override;
    void back_propagate() override;
    void gradient_descent() override;
	void clear_gradients() override;

	virtual const std::string  dump_weights() override { return "NOT IMPLEMENTED YET"; }
    virtual const std::string  dump_bias() override { return "NOT IMPLEMENTED YET"; }
    virtual const std::string  dump_activations() override { return "NOT IMPLEMENTED YET"; }

    const size_t count_layers() override { return m_layers.size(); }
	const layer_ptr get_layer_ptr( const size_t layer_idx ) override;
    void set_layer_ptr( const size_t layer_idx, const layer_ptr& l ) override;

protected:

    size_t m_training_samples;

	std::shared_ptr<tensor_solver_iface> m_solver;

    std::vector<std::shared_ptr<layer>> m_layers;
};

} /*namespace neurocl*/ } /*namespace convnet*/

#endif //NETWORK_CONVNET_H
