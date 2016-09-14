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

#include "network_factory.h"
#include "network_exception.h"

#include "mlp/network_manager.h"
#include "convnet/network_manager.h"

namespace neurocl {

std::shared_ptr<network_manager_interface> network_factory::build( const t_neural_impl& impl )
{
    switch( impl )
    {
    case NEURAL_IMPL_MLP:
        return mlp::network_manager::create( mlp::network_manager::MLP_IMPL_BNU_FAST );
    case NEURAL_IMPL_CONVNET:
        return convnet::network_manager::create();
    default:
        throw network_exception( "unmanaged neural implementation!" );
    }
}

} //namespace neurocl
