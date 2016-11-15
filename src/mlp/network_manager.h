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

#ifndef NETWORK_MANAGER_MLP_H
#define NETWORK_MANAGER_MLP_H

#include "common/network_manager_base.h"
#include "common/network_sample.h"

#include <boost/shared_array.hpp>

#include <vector>

namespace neurocl {

class network_factory;
class samples_manager;

namespace mlp {

class network_interface;
class network_file_handler;

class network_manager : public network_manager_base
{
public:

    enum class t_mlp_impl
    {
        MLP_IMPL_BNU_REF = 0,
		MLP_IMPL_BNU_FAST,
        MLP_IMPL_VEXCL
    };

private:

	friend network_factory;

	static std::shared_ptr<network_manager_interface> create( const t_mlp_impl& impl )
	{
		struct make_shared_enabler : public network_manager {
			make_shared_enabler( const t_mlp_impl& impl ) : network_manager( impl ) {}
		};
		return std::make_shared<make_shared_enabler>( impl );
	}

    network_manager( const t_mlp_impl& impl );

public:

	virtual ~network_manager() {}
};

} /*namespace neurocl*/ } /*namespace mlp*/

#endif //NETWORK_MANAGER_MLP_H
