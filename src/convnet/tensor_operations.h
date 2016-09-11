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

#ifndef TENSOR_OPERATIONS_H
#define TENSOR_OPERATIONS_H

#include "tensor.h"

namespace neurocl { namespace convnet {

class optimizer;

class tensor_operation
{
public:

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

    enum optimize_mode
    {
        optim_std = 0,
        optim_redux
    };

public:

    // returns aB (scalar product)
    static tensor scale( const float& val, const tensor& input );

    // returns A + B
    static tensor add( const tensor& inputA, const tensor& inputB );

    // returns A - B
    static tensor sub( const tensor& inputA, const tensor& inputB );

    // groups all submatrixes into a single one
    static tensor group( const tensor& input );

    // ungroups input matrix into multiple ones
    static void ungroup( const tensor& input, tensor& output );

    // returns A.B (standard product)
    static tensor elemul( const tensor& inputA, const tensor& inputB );

    // returns A.B (element product)
    static tensor mul( const tensor& inputA, const tensor& inputB );

    // returns A.B + C
    static tensor muladd( const tensor& inputA, const tensor& inputB, const tensor& inputC );

    // returns trans(A).B
    static tensor multrans1( const tensor& inputA, const tensor& inputB );

    // returns A.trans(B)
    static tensor multrans2( const tensor& inputA, const tensor& inputB );

    static void sig( tensor& input );

    static tensor d_sig( const tensor& input );

    template<kernel_mode km, pad_mode pm>
    static tensor convolve( const tensor& input, const tensor& filter, const int stride );

    template<kernel_mode km, pad_mode pm>
    static tensor convolve_add( const tensor& input, const tensor& filter, const int stride );

    static tensor subsample( const tensor& input, const size_t subsample );

    static tensor d_subsample( const tensor& input, const tensor& input_ref, const size_t subsample );

    static tensor uniform_sum( const tensor& input );

    template<optimize_mode om>
    static void optimize( const std::shared_ptr<optimizer>& optimizer, tensor& input, const tensor& deltas );
};

inline tensor operator*( const float& val, const tensor& t )
{
    return std::move( tensor_operation::scale( val, t ) );
};

inline tensor operator+( const tensor& t1, const tensor& t2 )
{
    return std::move( tensor_operation::add( t1, t2 ) );
};

inline tensor operator-( const tensor& t1, const tensor& t2 )
{
    return std::move( tensor_operation::sub( t1, t2 ) );
};

} /*namespace neurocl*/ } /*namespace convnet*/

#endif //TENSOR_OPERATIONS_H
