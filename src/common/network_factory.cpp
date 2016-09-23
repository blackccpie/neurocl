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
#include "network_config.h"

#include "mlp/network_manager.h"
#include "convnet/network_manager.h"

#include <boost/lexical_cast.hpp>

namespace neurocl {

std::istream& operator>> ( std::istream &input, network_factory::t_neural_impl& impl )
{
    std::string impl_string;
    input >> impl_string;

    if ( impl_string == "MLP" )
        impl = network_factory::NEURAL_IMPL_MLP;
    else if ( impl_string == "CONVNET" )
        impl = network_factory::NEURAL_IMPL_CONVNET;
    else
        input.setstate( std::ios_base::failbit );

    return input;
}

std::shared_ptr<network_manager_interface> network_factory::build()
{
    std::string str_impl = "undefined";

    try
    {
        const network_config& nc = network_config::instance();
        nc.update_mandatory( "implementation", str_impl );

        return build( boost::lexical_cast<t_neural_impl>( str_impl ) );
    }
    catch(...)
    {
        throw network_exception( "unmanaged neural implementation in configuration file : " + str_impl );
    }
}

std::shared_ptr<network_manager_interface> network_factory::build( const t_neural_impl& impl )
{
    switch( impl )
    {
    case NEURAL_IMPL_MLP:
        return mlp::network_manager::create( mlp::network_manager::MLP_IMPL_BNU_REF );
    case NEURAL_IMPL_CONVNET:
        return convnet::network_manager::create();
    default:
        throw network_exception( "unmanaged neural implementation!" );
    }
}

} //namespace neurocl
