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

#ifndef NETWORK_INTERFACE_H
#define NETWORK_INTERFACE_H

#include <iostream>
#include <vector>

namespace neurocl {

struct layer_size
{
    layer_size( const size_t& sX, const size_t& sY ) : sizeX( sX ), sizeY( sY ) {}

    const size_t sizeX;
    const size_t sizeY;

    const size_t size() const { return sizeX * sizeY; }
};

inline std::ostream& operator<< ( std::ostream& stream, const layer_size& size )
{
    stream << size.sizeX << "x" << size.sizeY;
    return stream;
}

class network_interface
{
public:

    // Convention : input layer is index 0
    virtual void add_layers_2d( const std::vector<layer_size>& layer_sizes ) = 0;

    virtual void set_input_sample(  const size_t& isample_size, const float* isample,
                                    const size_t& osample_size, const float* osample ) = 0;

    virtual void feed_forward() = 0;
    virtual void gradient_descent() = 0;

    virtual const float output() = 0;

    virtual const std::string dump_weights() = 0;
};

} //namespace neurocl

#endif //NETWORK_INTERFACE_H
