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

#include "network_exception.h"
#include "tensor.h"

#include <boost/numeric/ublas/matrix_proxy.hpp>

namespace neurocl {

inline float sigmoid( float x )
{
    return 1.f / ( 1.f + std::exp(-x) );
}

void _assert_same_depths1( const tensor& t1, const tensor& t2 )
{
    if ( t1.d1() != t2.d1() )
        throw network_exception( "inconsistent tensor number of feature maps" );
}

void _assert_same_sizes( const tensor& t1, const tensor& t2 )
{
    if ( ( t1.w() != t2.w() ) ||
        ( t1.h() != t2.h() ) ||
        ( t1.d1() != t2.d1() ) ||
        ( t1.d2() != t2.d2() ) )
        throw network_exception( "inconsistent tensor sizes" );
}

void tensor::_assert_same_size( const tensor& t )
{
    if ( ( m_width != t.w() ) ||
        ( m_height != t.h() ) ||
        ( m_depth1 != t.d1() ) ||
        ( m_depth2 != t.d2() ) )
        throw network_exception( "inconsistent tensor size" );
}

tensor tensor::operator +=( const tensor& other )
{
    _assert_same_size( other );

    tensor_foreach() {
        m_tensor_array[d1][d2] += other.c_m(d1,d2);
    }

    return *this;
}

tensor tensor::operator /( const float val )
{
    tensor_foreach() {
        m_tensor_array[d1][d2] /= val;
    }

    return *this;
}

tensor tensor_operation::elemul( const tensor& inputA, const tensor& inputB )
{
    using namespace boost::numeric::ublas;

    _assert_same_sizes( inputA, inputB );

    tensor output;
    output.resize( inputA );

    tensor_foreach_p( inputA.d1(), inputA.d2() ) {
        output.m(d1,d2) = element_prod( inputA.c_m(d1,d2), inputB.c_m(d1,d2) );
    }

    return output;
}

tensor tensor_operation::mul( const tensor& inputA, const tensor& inputB )
{
    using namespace boost::numeric::ublas;

    _assert_same_sizes( inputA, inputB );

    tensor output;
    output.resize( inputA );

    tensor_foreach_p( inputA.d1(), inputA.d2() ) {
        output.m(d1,d2) = prod( inputA.c_m(d1,d2), inputB.c_m(d1,d2) );
    }

    return output;
}

tensor tensor_operation::muladd( const tensor& inputA, const tensor& inputB, const tensor& inputC )
{
    using namespace boost::numeric::ublas;

    _assert_same_sizes( inputA, inputB );
    _assert_same_sizes( inputA, inputC );

    tensor output;
    output.resize( inputA );

    tensor_foreach_p( inputA.d1(), inputA.d2() ) {
        output.m(d1,d2) = prod( inputA.c_m(d1,d2), inputB.c_m(d1,d2) )
                + inputC.c_m(d1,d2);
    }

    return output;
}

tensor tensor_operation::multrans( const tensor& inputA, const tensor& inputB )
{
    using namespace boost::numeric::ublas;

    _assert_same_sizes( inputA, inputB );

    tensor output;
    output.resize( inputA );

    tensor_foreach_p( inputA.d1(), inputA.d2() ) {
        output.m(d1,d2) = prod( trans( inputA.c_m(d1,d2) ), inputB.c_m(d1,d2) );
    }

    return output;
}

void tensor_operation::sig( tensor& input )
{
    tensor_foreach_p( input.d1(), input.d2() ) {
        std::for_each(  input.m(d1,d2).data().begin(),
                        input.m(d1,d2).data().end(),
                        std::ptr_fun( sigmoid ) );
    }
}

tensor tensor_operation::d_sig( const tensor& input )
{
    using namespace boost::numeric::ublas;

    tensor output;
    output.resize( input );

    tensor_foreach_p( input.d1(), input.d2() ) {
        const matrixF& mat = input.c_m(d1,d2);
        output.m(d1,d2) = element_prod(
            mat,
            ( scalar_matrix<float>( mat.size1() * mat.size2(), 1.f ) - mat )
        );
    }

    return output;
}

struct flipper
{
    flipper( int sx, int sy ) { m_flipped = matrixF(sx,sy); }
    const matrixF& flipped( const matrixF& in )
    { m_flipped = in; std::reverse( m_flipped.data().begin(), m_flipped.data().end() ); return m_flipped; }
    matrixF m_flipped;
};

template <>
tensor tensor_operation::convolve_add<tensor_operation::kernel_flip,tensor_operation::pad_valid>(
    const tensor& input, const tensor& filter, const int stride )
{
    using namespace boost::numeric::ublas;

    _assert_same_depths1( input, filter );

    tensor output;

    auto stepsX = input.w() - filter.w() + 1;
    auto stepsY = input.h() - filter.h() + 1;

    output.resize( stepsX, stepsY, 1, filter.d2() );

    flipper f( filter.w(), filter.h() );

    // NOTE : tricky thing is that filter tensor replication level (filter.d2)
    // is equal to input tensor feature maps level (prev_layer.d1);
    // whereas filter feature maps level is equal to output tensor feature maps level

    for ( auto d1 = 0; d1 < filter.d1(); d1++ )
    {
        for ( auto d2 = 0; d2 < filter.d2(); d2++ )
        {
            for ( auto j=0; j<stepsY; j++ )
            {
                for ( auto i=0; i<stepsX; i++ )
                {
                    matrixF conv = element_prod( f.flipped( filter.c_m(d1,d2) ),
                        project( input.c_m(1,d1),
                            range( i, i+filter.w() ),
                            range( j, j+filter.h() ) ) );

                    output.m(1,d2)(i,j) += std::accumulate( conv.data().begin(), conv.data().end(), 0.f );
                }
            }
        }
    }

    return output;
}

template <>
tensor tensor_operation::convolve_add<tensor_operation::kernel_std,tensor_operation::pad_same>(
    const tensor& input, const tensor& filter, const int stride )
{
    tensor output;

    // TODO-CNN

    return output;
}

} //namespace neurocl
