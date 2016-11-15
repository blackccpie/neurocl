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

#include "network_vexcl.h"
#include "network_bnu_ref.h"
#include "network_manager.h"
#include "network_file_handler.h"

#ifdef SIMD_ENABLED
    #include "network_bnu_fast.h"
#endif

#include "common/network_exception.h"

//#define TRAIN_CHRONO

namespace neurocl { namespace mlp {

network_manager::network_manager( const t_mlp_impl& impl )
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
        m_net = std::make_shared<network_vexcl>();
        break;
    default:
        throw network_exception( "unmanaged mlp implementation!" );
    }

    m_net_file_handler =
        std::make_shared<network_file_handler>( std::static_pointer_cast<network_interface>( m_net ) );
}

} /*namespace neurocl*/ } /*namespace mlp*/
