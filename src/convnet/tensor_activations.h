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

#ifndef TENSOR_ACTIVATIONS_H
#define TENSOR_ACTIVATIONS_H

#include "tensor_operations.h"

namespace neurocl { namespace convnet {

class tensor_activation
{
public:

    static void sig( tensor& input )
    {
        tensor_foreach_p( input.d1(), input.d2() ) {
            std::for_each(  input.m(d1,d2).data().begin(),
                            input.m(d1,d2).data().end(),
                            []( float& a) { a = 1.f / ( 1.f + std::exp(-a) ); } );
        }
    }

    static tensor d_sig( const tensor& input )
    {
        using namespace boost::numeric::ublas;

        tensor output;
        output.resize( input );

        tensor_foreach_p( input.d1(), input.d2() ) {
            const matrixF& mat = input.c_m(d1,d2);
            output.m(d1,d2) = element_prod(
                mat,
                ( scalar_matrix<float>( mat.size1(), mat.size2(), 1.f ) - mat )
            );
        }

        return output;
    }

    static void relu( tensor& input )
    {
        tensor_foreach_p( input.d1(), input.d2() ) {
            std::for_each(  input.m(d1,d2).data().begin(),
                            input.m(d1,d2).data().end(),
                            []( float& a) { a = std::max( 0.f, a ); } );
        }
    }

    static tensor d_relu( const tensor& input )
    {
        using namespace boost::numeric::ublas;

        tensor output;
        output.resize( input );

        tensor_foreach_p( input.d1(), input.d2() ) {
            output.m(d1,d2) = input.c_m(d1,d2);
            std::for_each(  output.m(d1,d2).data().begin(),
                            output.m(d1,d2).data().end(),
                            []( float& a) { a = ( a > 0.f ) * 1.f; } );
        }

        return output;
    }

    static void softmax( tensor& input )
    {
        tensor_foreach_p( input.d1(), input.d2() ) {
            std::for_each(  input.m(d1,d2).data().begin(),
                            input.m(d1,d2).data().end(),
                            []( float& a) { a = std::max( 0.f, a ); } ); //TODO!!!!!!!!
        }
    }

    static tensor d_softmax( const tensor& input )
    {
        using namespace boost::numeric::ublas;

        tensor output;
        output.resize( input );

        tensor_foreach_p( input.d1(), input.d2() ) {
            output.m(d1,d2) = input.c_m(d1,d2);
            std::for_each(  output.m(d1,d2).data().begin(),
                            output.m(d1,d2).data().end(),
                            []( float& a) { a = ( a > 0.f ) * 1.f; } ); //TODO!!!!!!!!
        }

        return output;
    }
};

} /*namespace neurocl*/ } /*namespace convnet*/

#endif //TENSOR_ACTIVATIONS_H
