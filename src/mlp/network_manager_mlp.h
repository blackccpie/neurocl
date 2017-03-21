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

#include "network_bnu_ref.h"
#include "network_file_handler.h"

#ifdef SIMD_ENABLED
    #include "network_bnu_fast.h"
#endif

#ifdef VEXCL_ENABLED
    #include "network_vexcl.h"
#endif

#include "common/network_exception.h"
#include "common/network_manager.h"

namespace neurocl { namespace mlp {

class network_manager_mlp : public network_manager
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
		struct make_shared_enabler : public network_manager_mlp {
			make_shared_enabler( const t_mlp_impl& impl ) : network_manager_mlp( impl ) {}
		};
		return std::make_shared<make_shared_enabler>( impl );
	}

    network_manager_mlp( const t_mlp_impl& impl )
    {
        switch( impl )
        {
        case t_mlp_impl::MLP_IMPL_BNU_REF:
            m_net = std::make_shared<network_bnu_ref>();
            break;
        case t_mlp_impl::MLP_IMPL_BNU_FAST:
    #ifdef SIMD_ENABLED
            m_net = std::make_shared<network_bnu_fast>();
    #else
            throw network_exception( "unmanaged mlp implementation (simd disabled)!" );
    #endif
            break;
        case t_mlp_impl::MLP_IMPL_VEXCL:
    #ifdef VEXCL_ENABLED
            m_net = std::make_shared<network_vexcl>();
    #else
            throw network_exception( "unmanaged mlp implementation (opencl disabled)!" );
    #endif
            break;
        default:
            throw network_exception( "unmanaged mlp implementation!" );
        }

        m_net_file_handler = std::make_shared<network_file_handler>(
            std::static_pointer_cast<network_interface_mlp>( m_net ) );
    }

public:

	virtual ~network_manager_mlp() {}
};

} /*namespace neurocl*/ } /*namespace mlp*/

#endif //NETWORK_MANAGER_MLP_H
