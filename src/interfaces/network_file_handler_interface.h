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

#ifndef NETWORK_FILE_HANDLER_INTERFACE_H
#define NETWORK_FILE_HANDLER_INTERFACE_H

#include <string>

namespace neurocl {

class network_file_handler_interface
{
public:

    //! load network topology text file
    virtual void load_network_topology( const std::string& topology_path ) = 0;
    //! load network weights binary file
    virtual void load_network_weights( const std::string& weights_path ) = 0;
    //! save updated network weights
    virtual void save_network_weights() = 0;
};

} /*namespace neurocl*/

#endif //NETWORK_FILE_HANDLER_INTERFACE_H
