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

#include "convnet/tensor_operations.h"
#include "convnet/tensor_activations.h"

#include <iostream>

int main( int argc, char *argv[] )
{
    using nto = neurocl::convnet::tensor_operation;

    namespace nta = neurocl::convnet::tensor_activations;

    neurocl::convnet::tensor A,B,C,Res,Comp,O,ResO,CompO;

    A.resize(4,4,1,1);
    A.uniform_fill( 1.f );

    B.resize(4,4,1,1);
    B.uniform_fill( 1.f );

    C.resize(4,4,1,1);
    C.uniform_fill( -4.f );

    ResO.resize(4,1,1,1);
    CompO.resize(4,1,1,1);

    Comp.resize(4,4,1,1);
    Comp.uniform_fill( 0.f );

    Res.resize(4,4,1,1);

    // MULADD

    Res = nto::muladd( A, B, C );

    std::cout << "muladd test : " << ( ( Res == Comp ) ? "PASSED" : "FAILED" ) << std::endl;

    // ELEMUL

    A.uniform_fill( 2.f );
    B.uniform_fill( 2.f );
    Comp.uniform_fill( 4.f );

    Res = nto::elemul( A, B );

    std::cout << "elemul test : " << ( ( Res == Comp ) ? "PASSED" : "FAILED" ) << std::endl;

    // SUBSTRACT

    Comp.uniform_fill( 0.f );

    Res = A - B;

    std::cout << "substraction test : " << ( ( Res == Comp ) ? "PASSED" : "FAILED" ) << std::endl;

    // SIG

    Res.uniform_fill( 2.f );

    Comp.uniform_fill( 1.f / ( 1.f + std::exp(-2.f) ) );

    nta::sigmoid::f( Res );

    std::cout << "sig test : " << ( ( Res == Comp ) ? "PASSED" : "FAILED" ) << std::endl;

    // D_SIG

    A.uniform_fill( 2.f );
    Comp.uniform_fill( -2.f );

    Res = nta::sigmoid::d_f( A );

    std::cout << "d_sig test : " << ( ( Res == Comp ) ? "PASSED" : "FAILED" ) << std::endl;

    // RELU

    Res.uniform_fill( 1.f );
    Comp.uniform_fill( 1.f );

    nta::relu::f( Res );

    std::cout << "relu test1 : " << ( ( Res == Comp ) ? "PASSED" : "FAILED" ) << std::endl;

    Res.uniform_fill( -1.f );
    Comp.uniform_fill( 0.f );

    nta::relu::f( Res );

    std::cout << "relu test2 : " << ( ( Res == Comp ) ? "PASSED" : "FAILED" ) << std::endl;

    // D_RELU

    A.uniform_fill( 2.f );
    Comp.uniform_fill( 1.f );

    Res = nta::relu::d_f( A );

    std::cout << "d_relu test1 : " << ( ( Res == Comp ) ? "PASSED" : "FAILED" ) << std::endl;

    A.uniform_fill( -1.f );
    Comp.uniform_fill( 0.f );

    Res = nta::relu::d_f( A );

    std::cout << "d_relu test2 : " << ( ( Res == Comp ) ? "PASSED" : "FAILED" ) << std::endl;

    // SOFTMAX

    float out[4] = { 1.f, 2.f, 3.f, 4.f };
    ResO.fill( 0, 0, 4, out );

    std::for_each(  out,
                    out+4,
                    []( float& a) { a = std::exp(a-4.f) / ( std::exp(-3.f) + std::exp(-2.f) + std::exp(-1.f) + std::exp(0.f) ); } );
    CompO.fill( 0, 0, 4, out );

    nta::softmax::f( ResO );

    std::cout << "softmax test : " << ( ( ResO == CompO ) ? "PASSED" : "FAILED" ) << std::endl;

    // D_SOFTMAX

    // NOT IMPLEMENTED YET

    // INCREMENT

    A.uniform_fill( 1.f );
    Res.uniform_fill( 1.f );
    Comp.uniform_fill( 2.f );

    Res += A;

    std::cout << "increment test : " << ( ( Res == Comp ) ? "PASSED" : "FAILED" ) << std::endl;

    // DECREMENT

    Res.uniform_fill( 1.f );
    Comp.uniform_fill( 0.f );

    Res -= A;

    std::cout << "decrement test : " << ( ( Res == Comp ) ? "PASSED" : "FAILED" ) << std::endl;

    // SCALAR MULTIPLIER

    A.uniform_fill( 2.f );

    Comp.uniform_fill( 4.f );

    Res = 2.f * A;

    std::cout << "scalar multiplier test : " << ( ( Res == Comp ) ? "PASSED" : "FAILED" ) << std::endl;

    // SCALAR DIVIDER

    Comp.uniform_fill( 1.f );

    Res = A / 2.f;

    std::cout << "scalar divider test : " << ( ( Res == Comp ) ? "PASSED" : "FAILED" ) << std::endl;

    // UNIFORM SUM

    Comp.uniform_fill( 16.f );

    Res = nto::uniform_sum( A );

    std::cout << "uniform sum test : " << ( ( Res == Comp ) ? "PASSED" : "FAILED" ) << std::endl;

    // MULTRANS1

    A.resize(1,4,1,1);
    B.resize(1,4,1,1);

    A.uniform_fill( 2.f );
    B.uniform_fill( 2.f );
    Comp.uniform_fill( 4.f );

    Res = nto::multrans1( A, B );

    std::cout << "multrans1 test : " << ( ( Res == Comp ) ? "PASSED" : "FAILED" ) << std::endl;

    // MULTRANS2

    A.resize(4,1,1,1);
    B.resize(4,1,1,1);

    A.uniform_fill( 2.f );
    B.uniform_fill( 2.f );
    Comp.uniform_fill( 4.f );

    Res = nto::multrans2( A, B );

    std::cout << "multrans2 test : " << ( ( Res == Comp ) ? "PASSED" : "FAILED" ) << std::endl;

    // SUBSAMPLE

    A.resize(10,10,1,1);
    matrixF matA(10,10,0.f);
    for( auto i=0; i<10; i++ )
        for( auto j=0; j<10; j++ )
            if ( ( i%2 == 0 ) && ( j%2 == 0 ) )
                matA(i,j) = 1.f;
    A.fill(0,0,100,&matA.data()[0]);

    Comp.resize(5,5,1,1);
    Comp.uniform_fill( 1.f );

    Res.resize(5,5,1,1);

    Res = nto::subsample( A, 2 );

    std::cout << "subsample test : " << ( ( Res == Comp ) ? "PASSED" : "FAILED" ) << std::endl;

    // D_SUBSAMPLE

    A.resize(5,5,1,1);
    A.uniform_fill( 1.f );

    B.resize(10,10,1,1);
    matrixF matB(10,10);
    for( auto i=0; i<10; i++ )
        for( auto j=0; j<10; j++ )
            if ( ( i%2 == 0 ) && ( j%2 == 0 ) )
                matB(i,j) = 1.f;
            else
                matB(i,j) = 0.f;
    B.fill(0,0,100,&matB.data()[0]);

    Comp.resize(10,10,1,1);
    Comp = B;

    Res.resize(10,10,1,1);

    Res = nto::d_subsample( A, B, 2 );

    std::cout << "d_subsample test : " << ( ( Res == Comp ) ? "PASSED" : "FAILED" ) << std::endl;

    // BERNOULLI

    Res.resize(100,100,1,1);
    Comp.resize(100,100,1,1);

    nto::bernoulli( Res, 0.25f );
    nto::bernoulli( Comp, 0.25f );

    std::cout << "bernoulli test1 : " << ( !( Res == Comp ) ? "PASSED" : "FAILED" ) << std::endl;
    std::cout << "bernoulli test2 : " << ( ( ( Res.norm1() - 2500.f ) < 100.f ) ? "PASSED" : "FAILED" ) << std::endl;
    std::cout << "bernoulli test3 : " << ( ( ( Comp.norm1() - 2500.f ) < 100.f ) ? "PASSED" : "FAILED" ) << std::endl;

    // GROUP

    A.resize(5,5,1,2);
    A.uniform_fill( 2.5f );

    Comp.resize(50,1,1,1);
    Comp.uniform_fill( 2.5f );

    Res.resize(50,1,1,1);

    Res = nto::group( A );

    std::cout << "group test : " << ( ( Res == Comp ) ? "PASSED" : "FAILED" ) << std::endl;

    // UNGROUP

    A.resize(50,1,1,1);
    A.uniform_fill( 3.5f );

    Comp.resize(5,5,1,2);
    Comp.uniform_fill( 3.5f );

    Res.resize(5,5,1,2);

    nto::ungroup( A, Res );

    std::cout << "ungroup test : " << ( ( Res == Comp ) ? "PASSED" : "FAILED" ) << std::endl;

    // CONVOLVE ADD FORWARD FLIP/VALID

    A.resize(6,6,1,2);
    matA.resize(6,6);
    for( auto i=0; i<6; i++ )
        for( auto j=0; j<6; j++ )
            if ( i%3 == 0 )
                matA(i,j) = 2.f;
            else
                matA(i,j) = 1.f;
    A.fill(0,0,36,&matA.data()[0]);
    A.fill(0,1,36,&matA.data()[0]);

    B.resize(3,3,2,2);
    matB.resize(3,3);
    for( auto i=0; i<3; i++ )
        for( auto j=0; j<3; j++ )
                matB(i,j) = 1.f;
    matB(2,2) = 2.f;
    B.fill(0,0,9,&matB.data()[0]);
    B.fill(0,1,9,&matB.data()[0]);
    B.fill(1,0,9,&matB.data()[0]);
    B.fill(1,1,9,&matB.data()[0]);

    Comp.resize(4,4,1,2);
    matrixF matComp(4,4);
    for( auto i=0; i<4; i++ )
        for( auto j=0; j<4; j++ )
            if ( ( i == 0 ) || ( i == 3 ) )
                matComp(i,j) = 28.f;
            else
                matComp(i,j) = 26.f;
    Comp.fill(0,0,16,&matComp.data()[0]);
    Comp.fill(0,1,16,&matComp.data()[0]);

    Res.resize(6,6,1,2);

    Res = nto::convolve_add_forward<nto::kernel_mode::flip,nto::pad_mode::valid>( A, B, 1 );

    std::cout << "convolve_add_forward flip/valid test : " << ( ( Res == Comp ) ? "PASSED" : "FAILED" ) << std::endl;

    // CONVOLVE ADD BACKWARD STD/FULL

    A.resize(4,4,1,2);
    matA.resize(4,4);
    for( auto i=0; i<4; i++ )
        for( auto j=0; j<4; j++ )
            if ( i%3 == 0 )
                matA(i,j) = 2.f;
            else
                matA(i,j) = 1.f;
    A.fill(0,0,16,&matA.data()[0]);
    A.fill(0,1,16,&matA.data()[0]);

    Comp.resize(6,6,1,2);
    std::vector<float> vComp({ 8,12,16,16,8,4,
                        8,14,20,20,12,6,
                        10,18,26,26,16,8,
                        12,20,28,28,16,8,
                        6,12,18,18,12,6,
                        4,8,12,12,8,4});
    Comp.fill(0,0,36,&vComp[0]);
    Comp.fill(0,1,36,&vComp[0]);

    Res.resize(6,6,1,2);

    Res = nto::convolve_add_backward<nto::kernel_mode::std,nto::pad_mode::full>( A, B, 1 );

    std::cout << "convolve_add_backward std/full test : " << ( ( Res == Comp ) ? "PASSED" : "FAILED" ) << std::endl;

    // CONVOLVE UPDATE FLIP/VALID

    A.resize(6,6,1,2);
    matA.resize(6,6);
    for( auto i=0; i<6; i++ )
        for( auto j=0; j<6; j++ )
            if ( i%4 == 0 )
                matA(i,j) = 2.f;
            else
                matA(i,j) = 1.f;
    A.fill(0,0,36,&matA.data()[0]);
    A.fill(0,1,36,&matA.data()[0]);

    B.resize(4,4,1,2);
    matB.resize(4,4);
    for( auto i=0; i<4; i++ )
        for( auto j=0; j<4; j++ )
                matB(i,j) = 1.f;
    matB(0,0) = 2.f;
    B.fill(0,0,16,&matB.data()[0]);
    B.fill(0,1,16,&matB.data()[0]);

    Comp.resize(3,3,2,2);
    matComp.resize(3,3);
    for( auto i=0; i<3; i++ )
        for( auto j=0; j<3; j++ )
            if ( i == 0 )
                matComp(i,j) = 22.f;
            else
                matComp(i,j) = 21.f;
    Comp.fill(0,0,9,&matComp.data()[0]);
    Comp.fill(0,1,9,&matComp.data()[0]);
    Comp.fill(1,0,9,&matComp.data()[0]);
    Comp.fill(1,1,9,&matComp.data()[0]);

    Res.resize(3,3,2,2);

    Res = nto::convolve_update<nto::kernel_mode::std,nto::pad_mode::valid>( A, B, 1 );

    std::cout << "convolve_update flip/valid test : " << ( ( Res == Comp ) ? "PASSED" : "FAILED" ) << std::endl;

    /*std::cout << A.dump(0,0) << std::endl << std::endl;
    std::cout << B.dump(0,0) << std::endl << std::endl;
    std::cout << Comp.dump(0,0) << std::endl << std::endl;
    std::cout << Res.dump(0,0) << std::endl;*/

    return 0;
}
