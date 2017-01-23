/*
The MIT License

Copyright (c) 2015-2017 Albert Murienne

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

namespace neurocl { namespace convnet { namespace tensor_activations {

class sigmoid
{
public:

    static void f( tensor& input )
    {
        tensor_foreach_p( input.d1(), input.d2() ) {
            std::for_each(  input.m(d1,d2,{}).data().begin(),
                            input.m(d1,d2,{}).data().end(),
                            []( float& a ) { a = 1.f / ( 1.f + std::exp(-a) ); } );
        }
    }

    static tensor d_f( const tensor& input )
    {
        using namespace boost::numeric::ublas;

        tensor output;
        output.resize( input );

        tensor_foreach_p( input.d1(), input.d2() ) {
            const matrixF& mat = input.c_m(d1,d2,{});
            output.m(d1,d2,{}) = element_prod(
                mat,
                ( scalar_matrix<float>( mat.size1(), mat.size2(), 1.f ) - mat )
            );
        }

        return output;
    }
};

class tanh
{
public:

    static void f( tensor& input )
    {
        // As seen in Lecun's Efficient Backprop :
        // http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf

        tensor_foreach_p( input.d1(), input.d2() ) {
            std::for_each(  input.m(d1,d2,{}).data().begin(),
                            input.m(d1,d2,{}).data().end(),
                            []( float& a ) { a = 1.7159f * ::tanh(2.f*a/3.f); } );
        }
    }

    static tensor d_f( const tensor& input )
    {
        using namespace boost::numeric::ublas;

        tensor output;
        output.resize( input );

        tensor_foreach_p( input.d1(), input.d2() ) {
            const matrixF& mat = input.c_m(d1,d2,{});
            output.m(d1,d2,{}) = scalar_matrix<float>( mat.size1(), mat.size2(), 1.f ) - element_prod( mat, mat );
        }

        return output;
    }
};

class relu
{
public:

    static void f( tensor& input )
    {
        tensor_foreach_p( input.d1(), input.d2() ) {
            std::for_each(  input.m(d1,d2,{}).data().begin(),
                            input.m(d1,d2,{}).data().end(),
                            []( float& a ) { a = std::max( 0.f, a ); } );
        }
    }

    static tensor d_f( const tensor& input )
    {
        using namespace boost::numeric::ublas;

        tensor output;
        output.resize( input );

        tensor_foreach_p( input.d1(), input.d2() ) {
            output.m(d1,d2,{}) = input.c_m(d1,d2,{});
            std::for_each(  output.m(d1,d2,{}).data().begin(),
                            output.m(d1,d2,{}).data().end(),
                            []( float& a ) { a = ( a > 0.f ) * 1.f; } );
        }

        return output;
    }
};

class leaky_relu
{
public:

    static void f( tensor& input )
    {
        tensor_foreach_p( input.d1(), input.d2() ) {
            std::for_each(  input.m(d1,d2,{}).data().begin(),
                            input.m(d1,d2,{}).data().end(),
                            []( float& a ) { a = a > 0.f ? a : 0.01f * a; } );
        }
    }

    static tensor d_f( const tensor& input )
    {
        using namespace boost::numeric::ublas;

        tensor output;
        output.resize( input );

        tensor_foreach_p( input.d1(), input.d2() ) {
            output.m(d1,d2,{}) = input.c_m(d1,d2,{});
            std::for_each(  output.m(d1,d2,{}).data().begin(),
                            output.m(d1,d2,{}).data().end(),
                            []( float& a ) { a = a > 0.f ? 1.f : 0.01f; } );
        }

        return output;
    }
};

class softmax
{
public:

    static void f( tensor& input )
    {
        float alpha = std::numeric_limits<float>::min();
        tensor_foreach_p( input.d1(), input.d2() ) {
            std::for_each(  input.m(d1,d2,{}).data().begin(),
                            input.m(d1,d2,{}).data().end(),
                            [&alpha]( float& a) { if ( a > alpha ) alpha = a; } );
        }
        float denom = 0.f;
        tensor_foreach_p( input.d1(), input.d2() ) {
            std::for_each(  input.m(d1,d2,{}).data().begin(),
                            input.m(d1,d2,{}).data().end(),
                            [alpha,&denom]( float& a) { denom += /*1e-10 +*/ std::exp(a - alpha); } );
        }
        tensor_foreach_p( input.d1(), input.d2() ) {
            std::for_each(  input.m(d1,d2,{}).data().begin(),
                            input.m(d1,d2,{}).data().end(),
                            [alpha,denom]( float& a) { a = std::exp(a - alpha)/denom; } );
        }
    }

    static tensor d_f( const tensor& input, const tensor& prev_delta )
    {
		// TODO-CNN : still not a fully validated implementation

        using namespace boost::numeric::ublas;

        tensor output;
        output.resize( input );

        matrixF _mdf( output.w(), output.h() );

        tensor_foreach_p( output.d1(), output.d2() ) {
            const matrixF& _input = input.c_m(d1,d2,{});
            const matrixF& _prev_delta = prev_delta.c_m(d1,d2,{});
            matrixF& _output = output.m(d1,d2,{});
            size_t index1 = 0;
            for ( auto i=0; i<_output.size1(); i++ )
                for ( auto j=0; j<_output.size2(); j++ )
                {
                    size_t index2 = 0;
                    for ( auto p=0; p<_mdf.size1(); p++ )
                        for ( auto q=0; q<_mdf.size2(); q++ )
                        {
                            _mdf(p,q) = ( index1 == index2 ) ? _df( _input(p,q) ) : -_input(p,q) * _input(i,j);
                            ++index2;
                        }
                    _output( i, j ) = _sum( element_prod( _prev_delta, _mdf ) );
                    ++index1;
                }
        }

        return output;
    }

    static tensor d_f( const tensor& input )
    {
		// One hot gradient version MUST be used in conjunction with softmax cross entropy cost:
        // http://www.ics.uci.edu/~pjsadows/notes.pdf
		// http://peterroelants.github.io/posts/neural_network_implementation_intermezzo02
        tensor identity;
        identity.resize( input );
        identity.uniform_fill(1.f);
        return identity;
    }

private:

    static float _df( float y )
    {
        return y * ( 1.f - y );
    }

    static float _sum( const matrixF& mat )
    {
        return std::accumulate( mat.data().begin(), mat.data().end(), 0.f );
    }
};

} /*namespace neurocl*/ } /*namespace convnet*/ } /*namespace tensor_activations*/

#endif //TENSOR_ACTIVATIONS_H
