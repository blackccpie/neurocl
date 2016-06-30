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

template <>
void tensor_operation::convolve_add<tensor_operation::kernel_flip,tensor_operation::pad_valid>(
    const tensor& input, const tensor& filter, tensor& output, const int stride )
{
    using namespace boost::numeric::ublas;

    auto stepsX = input.w() - filter.w() + 1;
    auto stepsY = input.h() - filter.h() + 1;

    for ( auto d1 = 0; d1 < filter.d1(); d1++ )
    {
        for ( auto d2 = 0; d2 < filter.d2(); d2++ )
        {
        for ( auto j=0; j<stepsY; j++ )
            for ( auto i=0; i<stepsX; i++ )
            {
                matrixF conv = element_prod( filter.const_array(d1,d2),
                    project( input.const_array(d1,1),
                        range( i, i+filter.w() ),
                        range( j, j+filter.h() ) ) );

                output.array(d2,1)(i,j) += std::accumulate( conv.data().begin(), conv.data().end(), 0.f );
            }
        }
    }
}

template <>
void tensor_operation::convolve_add<tensor_operation::kernel_flip,tensor_operation::pad_full>(
    const tensor& input, const tensor& filter, tensor& output, const int stride )
{

}

template <>
void tensor_operation::convolve_add<tensor_operation::kernel_std,tensor_operation::pad_full>(
    const tensor& input, const tensor& filter, tensor& output, const int stride )
{

}

} //namespace neurocl
