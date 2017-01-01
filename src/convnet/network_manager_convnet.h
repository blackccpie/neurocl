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

#include "network_parallel.h"
#include "network_file_handler.h"

#include "common/network_manager.h"

namespace neurocl { namespace convnet {

class network_manager_convnet : public network_manager
{
public:

	enum class t_convnet_impl
    {
        CONVNET = 0,
		CONVNET_PARALLEL
    };

private:

	friend network_factory;

	static std::shared_ptr<network_manager_interface> create( const t_convnet_impl& impl )
	{
		struct make_shared_enabler : public network_manager_convnet {
			make_shared_enabler( const t_convnet_impl& impl ) : network_manager_convnet( impl ) {}
		};
		return std::make_shared<make_shared_enabler>( impl );
	}

    network_manager_convnet( const t_convnet_impl& impl )
	{
		switch( impl )
		{
		case t_convnet_impl::CONVNET:
	        m_net = std::make_shared<network>();
	        break;
		case t_convnet_impl::CONVNET_PARALLEL:
		    m_net = std::make_shared<network_parallel>();
		    break;
	    default:
	        throw network_exception( "unmanaged convnet implementation!" );
		}
		m_net = std::make_shared<network/*_parallel*/>();
    	m_net_file_handler = std::make_shared<network_file_handler>(
			std::static_pointer_cast<network_interface_convnet>( m_net ) );
	}

public:

	virtual ~network_manager_convnet() {}
};

} /*namespace neurocl*/ } /*namespace convnet*/

#endif //NETWORK_MANAGER_CONVNET_H
