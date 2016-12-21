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

#ifndef NETWORK_PARALLEL_CONVNET_H
#define NETWORK_PARALLEL_CONVNET_H

#include "network.h"

#include <memory>
#include <vector>

namespace neurocl {

class thread_pool;

namespace convnet {

class tensor_solver_iface;

class network_parallel final : public network_interface_convnet
{
public:

	network_parallel();
	virtual ~network_parallel();

    virtual void add_layers( const std::vector<layer_descr>& layers ) final;

	virtual void set_training( bool training ) final;

	virtual void set_input(  const size_t& in_size, const float* in ) final;
    virtual void set_output( const size_t& out_size, const float* out ) final;
    virtual const output_ptr output() final;

    virtual void feed_forward() final;
    virtual void back_propagate() final;
    virtual void gradient_descent() final;
	virtual void clear_gradients() final;
	virtual void gradient_check( const output_ptr& out_ref ) final;

	virtual const std::string  dump_weights() final { return "NOT IMPLEMENTED YET"; }
    virtual const std::string  dump_bias() final { return "NOT IMPLEMENTED YET"; }
    virtual const std::string  dump_activations() final { return "NOT IMPLEMENTED YET"; }

    virtual const size_t count_layers() final;
	virtual const layer_ptr get_layer_ptr( const size_t layer_idx ) final;
    virtual void set_layer_ptr( const size_t layer_idx, const layer_ptr& l ) final;

private:

	void _feed_back( network* net );

protected:

	size_t m_current_net;

	std::unique_ptr<thread_pool> m_thread_pool;
	std::shared_ptr<tensor_solver_iface> m_solver;
	std::vector<network> m_networks;
};

} /*namespace neurocl*/ } /*namespace convnet*/

#endif //NETWORK_CONVNET_H
