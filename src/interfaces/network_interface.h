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

#include <cstddef>

namespace neurocl {

struct output_ptr
{
    // Constructor with already allocated buffers
    output_ptr( const size_t& no, boost::shared_array<float> o )
        : num_outputs( no ), outputs( o ) {}
    // Constructor with allocation requests
    output_ptr( const size_t& nw )
        : num_outputs( nw )
    {
        outputs.reset( new float[num_outputs] );
    }
    // Copy constructor
    output_ptr( const output_ptr& l )
        : num_outputs( l.num_outputs ), outputs( l.outputs ) {}
    // Destructor
    ~output_ptr() {}

    size_t max_comp_idx() const
    {
        return std::distance( outputs.get(), std::max_element( outputs.get(), outputs.get() + num_outputs ) );
    }

    float max_comp_val() const
    {
        return *std::max_element( outputs.get(), outputs.get() + num_outputs );
    }

    const size_t num_outputs;
    boost::shared_array<float> outputs;
};

class network_interface
{
public:

    //! Set training flag
    virtual void set_training( bool training ) = 0;

    //! set input values
    virtual void set_input(  const size_t& in_size, const float* in ) = 0;
    //! set output values
    virtual void set_output( const size_t& out_size, const float* out ) = 0;

    //! feed forward
    virtual void feed_forward() = 0;
    //! back propagate
    virtual void back_propagate() = 0;
    //! apply gradient descent
    virtual void gradient_descent() = 0;
    //! clear gradients
    virtual void clear_gradients() = 0;
    //! gradient check
    virtual void gradient_check( const output_ptr& out_ref ) = 0;

    //! get output values
    virtual const output_ptr output() = 0;

    //! network parameters dump
    virtual const std::string dump_weights() = 0;
    virtual const std::string dump_bias() = 0;
    virtual const std::string dump_activations() = 0;
};

} /*namespace neurocl*/

#endif //NETWORK_INTERFACE_H
