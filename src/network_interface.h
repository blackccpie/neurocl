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

#include <boost/shared_array.hpp>

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

struct layer_ptr
{
    // Constructor with already allocated buffers
    layer_ptr( const size_t& nw, boost::shared_array<float> w, const size_t& nb, boost::shared_array<float> b )
        : num_weights( nw ), weights( w ), num_bias( nb ), bias( b ) {}
    // Constructor with allocation requests
    layer_ptr( const size_t& nw, const size_t& nb )
        : num_weights( nw ), weights( 0 ), num_bias( nb ), bias( 0 )
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

struct output_ptr
{
    // Constructor with already allocated buffers
    output_ptr( const size_t& no, boost::shared_array<float> o )
        : num_outputs( no ), outputs( o ) {}
    // Constructor with allocation requests
    output_ptr( const size_t& nw )
        : num_outputs( nw ), outputs( 0 )
    {
        outputs.reset( new float[num_outputs] );
    }
    // Copy constructor
    output_ptr( const output_ptr& l )
        : num_outputs( l.num_outputs ), outputs( l.outputs ) {}
    // Destructor
    ~output_ptr() {}

    const size_t num_outputs;
    boost::shared_array<float> outputs;
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

    virtual void set_input(  const size_t& in_size, const float* in ) = 0;
    virtual void set_output( const size_t& out_size, const float* out ) = 0;

    virtual void feed_forward() = 0;

    virtual void prepare_training() = 0;
    virtual void back_propagate() = 0;
    virtual void update_params() = 0;

    virtual const size_t count_layers() = 0;
    virtual const layer_ptr get_layer_ptr( const size_t layer_idx ) = 0;
    virtual void set_layer_ptr( const size_t layer_idx, const layer_ptr& layer ) = 0;

    virtual const output_ptr output() = 0;

    virtual const std::string dump_weights() = 0;
    virtual const std::string dump_activations() = 0;
};

} //namespace neurocl

#endif //NETWORK_INTERFACE_H
