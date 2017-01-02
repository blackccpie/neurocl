/*
The MIT License

Copyright (c) 2015-2017 Albert Murienne

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

#ifndef NETWORK_MANAGER_H
#define NETWORK_MANAGER_H

#include "interfaces/network_manager_interface.h"

#include "common/network_sample.h"

#include <vector>

namespace neurocl {

class samples_manager;

class network_interface;
class network_file_handler_interface;

class network_manager : public network_manager_interface
{
public:

	network_manager() : m_network_loaded( false ) {}
	virtual ~network_manager() {}

	//! load network topology & weights
    virtual void load_network( const std::string& topology_path, const std::string& weights_path ) override;
	//! save network weights
	virtual void save_network() override;

	//! Set training flag
    virtual void set_training( bool training, key_training ) final;
	//! train given single sample
    virtual void train( const sample& s, key_training ) override;

	//! mini-batch train a samples set
    virtual void batch_train(	const samples_manager& smp_manager,
								const size_t& epoch_size,
								const size_t& batch_size,
								t_progress_fct progress_fct = t_progress_fct() ) override;

    //! prepare training epoch
    virtual void prepare_training_epoch() override;
    //! finalize training epoch
    virtual void finalize_training_epoch() override;
	//! compute network output
    virtual void compute_output( sample& s ) override;

	//! gradient check
	virtual void gradient_check( const sample& s ) override;

	//! dump network parameters
	virtual void dump_weights() override;
    virtual void dump_bias() override;
    virtual void dump_activations() override;

protected:

	void _assert_loaded();

private:

    void _train_single( const sample& s );
    void _train_batch( const std::vector<sample>& training_set );

private:

	bool m_network_loaded;

protected:

    std::shared_ptr<network_interface> m_net;
    std::shared_ptr<network_file_handler_interface> m_net_file_handler;
};

} /*namespace neurocl*/

#endif //NETWORK_MANAGER_H
