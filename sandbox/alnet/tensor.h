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

#ifndef TENSOR_H
#define TENSOR_H

#include <boost/multi_array.hpp>
#include <boost/numeric/ublas/vector.hpp>

#include <memory>

typedef typename boost::numeric::ublas::vector<float> vectorF;
typedef typename boost::multi_array<vectorF,2> mvector2F;

namespace neurocl {

class optimizer;

class tensor
{
public:
    tensor() {}
    virtual ~tensor() {}

    void resize( const size_t vec_size, const size_t depth1, const size_t depth2 )
    {
        m_tensor_array.resize( boost::extents[depth1][depth2] );
    }

private:

    mvector2F m_tensor_array;
};

struct tensor_operation
{
    enum kernel_mode
    {
        kernel_std = 0,
        kernel_flip
    };

    enum pad_mode
    {
        pad_valid = 0,
        pad_same,
        pad_full
    };

    template<kernel_mode km, pad_mode pm>
    static void convolve_add(
        const tensor& input, const tensor& filter, tensor& output, const int stride );

    static void optimize( std::shared_ptr<optimizer> optimizer, tensor& input, tensor& deltas )
    {

    }

    static void relu( tensor& input )
    {

    }

    static void d_relu( tensor& input, const tensor& output )
    {

    }
};

} //namespace neurocl

#endif //TENSOR_H
