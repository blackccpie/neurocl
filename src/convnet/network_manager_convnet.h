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

#ifndef NETWORK_MANAGER_CONVNET_H
#define NETWORK_MANAGER_CONVNET_H

#include "network.h"
#include "network_file_handler.h"

#include "common/network_manager.h"

namespace neurocl { namespace convnet {

class network_manager_convnet : public network_manager
{
private:

	friend network_factory;

	static std::shared_ptr<network_manager_interface> create()
	{
		struct make_shared_enabler : public network_manager_convnet {};
		return std::make_shared<make_shared_enabler>();
	}

    network_manager_convnet()
	{
		m_net = std::make_shared<network>();
    	m_net_file_handler = std::make_shared<network_file_handler>(
			std::static_pointer_cast<network_interface_convnet>( m_net ) );
	}

public:

	virtual ~network_manager_convnet() {}
};

} /*namespace neurocl*/ } /*namespace convnet*/

#endif //NETWORK_MANAGER_CONVNET_H
