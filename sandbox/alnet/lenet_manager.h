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

#ifndef LENET_MANAGER_H
#define LENET_MANAGER_H

#include "network_sample.h"

#include <boost/function.hpp>
#include <boost/shared_array.hpp>

#include <vector>

namespace neurocl {

class lenet_interface;
class samples_manager;
class lenet_file_handler;

class lenet_manager
{
public:

	typedef boost::function<void(int)> t_progress_fct;

public:

    lenet_manager();
	virtual ~lenet_manager() {}

    void load_network( const std::string& topology_path, const std::string& weights_path );
    void save_network();

    void train( const sample& s );
    void train( const std::vector<sample>& training_set );
    void batch_train(	const samples_manager& smp_manager,
						const size_t& epoch_size,
						const size_t& batch_size,
						t_progress_fct progress_fct = t_progress_fct() );

    // prepare gradient descent
    void prepare_training_iteration();
    // finalize gradient descent
    void finalize_training_iteration();

    void compute_output( sample& s );

private:

	void _assert_loaded();
    void _train( const sample& s );

private:

	bool m_network_loaded;

    std::shared_ptr<lenet_interface> m_net;
    std::shared_ptr<lenet_file_handler> m_net_file_handler;
};

} //namespace neurocl

#endif //LENET_MANAGER_H
