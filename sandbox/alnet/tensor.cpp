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

#include "tensor.h"

#include <boost/numeric/ublas/matrix_proxy.hpp>

namespace neurocl {

inline float sigmoid( float x )
{
    return 1.f / ( 1.f + std::exp(-x) );
}

tensor tensor_operation::muladd( const tensor& inputA, const tensor& inputB, const tensor& inputC )
{
    using namespace boost::numeric::ublas;

    tensor output;

    for ( auto d1 = 0; d1 < inputA.d1(); d1++ )
    {
        for ( auto d2 = 0; d2 < inputA.d2(); d2++ )
        {
            output.m(d1,d2) = prod( inputA.c_m(d1,d2), inputB.c_m(d1,d2) )
                + inputC.c_m(d1,d2);
        }
    }

    return output;
}

void tensor_operation::relu( tensor& input )
{
    for ( auto d1 = 0; d1 < input.d1(); d1++ )
    {
        for ( auto d2 = 0; d2 < input.d2(); d2++ )
        {
            std::for_each(  input.m(d1,d2).data().begin(),
                            input.m(d1,d2).data().end(),
                            std::ptr_fun( sigmoid ) );
        }
    }
}

void tensor_operation::d_relu( tensor& input, const tensor& output )
{

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

    tensor output;

    auto stepsX = input.w() - filter.w() + 1;
    auto stepsY = input.h() - filter.h() + 1;

    output.resize( stepsX, stepsY, 1, input.d2() );

    flipper f( filter.w(), filter.h() );

    for ( auto d1 = 0; d1 < filter.d1(); d1++ )
    {
        for ( auto d2 = 0; d2 < filter.d2(); d2++ )
        {
            for ( auto j=0; j<stepsY; j++ )
            {
                for ( auto i=0; i<stepsX; i++ )
                {
                    matrixF conv = element_prod( f.flipped( filter.c_m(d1,d2) ),
                        project( input.c_m(d1,1),
                            range( i, i+filter.w() ),
                            range( j, j+filter.h() ) ) );

                    output.m(d2,1)(i,j) += std::accumulate( conv.data().begin(), conv.data().end(), 0.f );
                }
            }
        }
    }

    return output;
}

template <>
tensor tensor_operation::convolve_add<tensor_operation::kernel_flip,tensor_operation::pad_full>(
    const tensor& input, const tensor& filter, const int stride )
{
    tensor output;

    return output;
}

template <>
tensor tensor_operation::convolve_add<tensor_operation::kernel_std,tensor_operation::pad_full>(
    const tensor& input, const tensor& filter, const int stride )
{
    tensor output;

    return output;
}

} //namespace neurocl
