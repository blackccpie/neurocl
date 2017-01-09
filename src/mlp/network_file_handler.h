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

#ifndef NETWORK_FILE_HANDLER_MLP_H
#define NETWORK_FILE_HANDLER_MLP_H

#include "interfaces/network_file_handler_interface.h"

#include "common/layer_storage.h"

#include <boost/shared_ptr.hpp>

namespace neurocl { namespace mlp {

class network_interface_mlp;

class network_file_handler : public network_file_handler_interface
{
public:

    network_file_handler( const std::shared_ptr<network_interface_mlp>& net );
    virtual ~network_file_handler();

    virtual void load_network_topology( const std::string& topology_path ) final;
    virtual void load_network_weights( const std::string& weights_path ) final;
    virtual void save_network_weights() final;

private:

    size_t m_layers;

    std::string m_weights_path;

    std::shared_ptr<network_interface_mlp> m_net;
};

} /*namespace neurocl*/ } /*namespace mlp*/

#endif //NETWORK_FILE_HANDLER_MLP_H
