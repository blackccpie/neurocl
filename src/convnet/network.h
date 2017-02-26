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

#include "network_interface_convnet.h"

#include <atomic>
#include <memory>
#include <vector>

namespace neurocl { namespace convnet {

class layer;
class tensor_solver_iface;

class network final : public network_interface_convnet
{
public:

	network();
	virtual ~network();

    virtual void add_layers( const std::vector<layer_descr>& layers ) final override;

    virtual void set_training( bool training ) final override;

	virtual void set_input(  const size_t& in_size, const float* in ) final override;
    virtual void set_output( const size_t& out_size, const float* out ) final override;
    virtual const output_ptr output() final override;

    virtual void feed_forward() final override;
    virtual void back_propagate() final override;
    virtual void gradient_descent() final override;
	virtual void clear_gradients() final override;
	virtual void gradient_check( const output_ptr& out_ref ) final override;
    virtual float loss() final override;

	virtual const std::string  dump_weights() final override { return "NOT IMPLEMENTED YET"; }
    virtual const std::string  dump_bias() final override { return "NOT IMPLEMENTED YET"; }
    virtual const std::string  dump_activations() final override { return "NOT IMPLEMENTED YET"; }

	void dump_image_features();

    virtual const size_t count_layers() final override { return m_layers.size(); }
	virtual const layer_ptr get_layer_ptr( const size_t layer_idx ) final override;
    virtual void set_layer_ptr( const size_t layer_idx, const layer_ptr& l ) final override;

protected:

    static std::atomic_size_t m_training_samples;

	std::shared_ptr<tensor_solver_iface> m_solver;

    std::vector<std::shared_ptr<layer>> m_layers;
};

} /*namespace neurocl*/ } /*namespace convnet*/

#endif //NETWORK_CONVNET_H
