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

#include "common/export.h"

#include "tensor.h"

namespace neurocl { namespace convnet {

class tensor_solver_iface;

class NEUROCL_PUBLIC tensor_operation
{
public:

    enum class kernel_mode
    {
        std = 0,
        flip
    };

    enum class pad_mode
    {
        valid = 0,
        same,
        full
    };

    enum class optimize_mode
    {
        std = 0,
        redux
    };

public:

    // returns aB (scalar product)
    static tensor scale( const float& val, const tensor& input );

    // returns scalar_mat(a) + B
    static tensor plus( const float& val, const tensor& input );

    // returns scalar_mat(a) - B
    static tensor minus( const float& val, const tensor& input );

    // returns A + B
    static tensor add( const tensor& inputA, const tensor& inputB );

    // returns A - B
    static tensor sub( const tensor& inputA, const tensor& inputB );

    // groups all submatrixes into a single one
    static tensor group( const tensor& input );

    // ungroups input matrix into multiple ones
    static void ungroup( const tensor& input, tensor& output );

    // returns A/B (element division)
    static tensor elediv( const tensor& inputA, const tensor& inputB );

    // returns A.B (element product)
    static tensor elemul( const tensor& inputA, const tensor& inputB );

    // returns A.B (standard product)
    static tensor mul( const tensor& inputA, const tensor& inputB );

    // returns A.B + C
    static tensor muladd( const tensor& inputA, const tensor& inputB, const tensor& inputC );

    // returns trans(A).B
    static tensor multrans1( const tensor& inputA, const tensor& inputB );

    // returns A.trans(B)
    static tensor multrans2( const tensor& inputA, const tensor& inputB );

    // returns element wise square root
    static tensor sqrt( const tensor& input );

    template<kernel_mode km, pad_mode pm>
    static tensor convolve_add_forward( const tensor& input, const tensor& filter, const int stride );

    template<kernel_mode km, pad_mode pm>
    static tensor convolve_add_backward( const tensor& input, const tensor& filter, const int stride );

    template<kernel_mode km, pad_mode pm>
    static tensor convolve_update( const tensor& input, const tensor& filter, const int stride );

    static tensor subsample( const tensor& input, const size_t subsample );

    static tensor d_subsample( const tensor& input, const tensor& input_ref, const size_t subsample );

    static tensor uniform_sum( const tensor& input );

    static void bernoulli( tensor& input, const float p );

    static tensor binary_operator( const tensor& inputA, const tensor& inputB, std::function<float (const float&,const float&)> op );

    template<optimize_mode om>
    static void optimize( const std::shared_ptr<tensor_solver_iface>& solver, tensor* input, tensor** input_cache, const tensor* deltas );
};

inline tensor operator*( const float& val, const tensor& t )
{
    return tensor_operation::scale( val, t );
}

inline tensor operator+( const float& val, const tensor& t )
{
    return tensor_operation::plus( val, t );
}

inline tensor operator-( const float& val, const tensor& t )
{
    return tensor_operation::minus( val, t );
}

inline tensor operator+( const tensor& t1, const tensor& t2 )
{
    return tensor_operation::add( t1, t2 );
}

inline tensor operator-( const tensor& t1, const tensor& t2 )
{
    return tensor_operation::sub( t1, t2 );
}

inline tensor operator*( const tensor& t1, const tensor& t2 )
{
    return tensor_operation::elemul( t1, t2 );
}

inline tensor operator/( const tensor& t1, const tensor& t2 )
{
    return tensor_operation::elediv( t1, t2 );
}

} /*namespace neurocl*/ } /*namespace convnet*/

#endif //TENSOR_OPERATIONS_H
