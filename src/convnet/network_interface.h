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

#include "common/network_base_interface.h"

#include <iostream>
#include <vector>

namespace neurocl { namespace convnet {

enum layer_type
{
    INPUT_LAYER = 0,
    CONV_LAYER,
    POOL_LAYER,
    FULL_LAYER,
    OUTPUT_LAYER
};

struct layer_descr
{
    layer_descr( const layer_type& t, const size_t& sX, const size_t& sY, const size_t& sZ, const size_t& sF )
        : type( t ), sizeX( sX ), sizeY( sY ), sizeZ( sZ ), sizeF( sF ) {}

    const layer_type type;

    const size_t sizeX;
    const size_t sizeY;
    const size_t sizeZ;
    const size_t sizeF; // optional filter size

    const size_t size() const { return sizeX * sizeY * sizeZ; }
};

struct layer_ptr
{
    // Constructor with already allocated buffers
    layer_ptr( const size_t& nw, boost::shared_array<float> w, const size_t& nb, boost::shared_array<float> b )
        : num_weights( nw ), weights( w ), num_bias( nb ), bias( b ) {}
    // Constructor with allocation requests
    layer_ptr( const size_t& nw, const size_t& nb )
        : num_weights( nw ), num_bias( nb )
    {
        weights.reset( new float[num_weights] );
        bias.reset( new float[num_bias] );
    }
    // Copy constructor
    layer_ptr( const layer_ptr& l )
        : num_weights( l.num_weights ), weights( l.weights ), num_bias( l.num_bias ), bias( l.bias ) {}
    // Destructor
    ~layer_ptr() {}

    const size_t num_weights;
    boost::shared_array<float> weights;
    const size_t num_bias;
    boost::shared_array<float> bias;
};

inline std::ostream& operator<< ( std::ostream& stream, const layer_descr& layer )
{
    // TODO-CNN : dump layer type string
    stream << layer.sizeX << "x" << layer.sizeY << "x" << layer.sizeZ;
    return stream;
}

class network_interface : public network_base_interface
{
public:

    // Convention : input layer is index 0
    virtual void add_layers( const std::vector<layer_descr>& layers ) = 0;

    virtual const size_t count_layers() = 0;
    virtual const layer_ptr get_layer_ptr( const size_t layer_idx ) = 0;
    virtual void set_layer_ptr( const size_t layer_idx, const layer_ptr& l ) = 0;
};

} /*namespace neurocl*/ } /*namespace convnet*/

#endif //NETWORK_INTERFACE_H
