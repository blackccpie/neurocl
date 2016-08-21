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

#include "alnet/tensor.h"

#include <iostream>

int main( int argc, char *argv[] )
{
    using nto = neurocl::tensor_operation;

    neurocl::tensor A,B,C,Res,Comp;

    A.resize(4,4,1,1);
    A.fill( 1.f );

    B.resize(4,4,1,1);
    B.fill( 1.f );

    C.resize(4,4,1,1);
    C.fill( -4.f );

    Comp.resize(4,4,1,1);
    Comp.fill( 0.f );

    Res.resize(4,4,1,1);

    // MULADD

    Res = nto::muladd( A, B, C );

    std::cout << "muladd test : " << ( ( Res == Comp ) ? "PASSED" : "FAILED" ) << std::endl;

    // ELEMUL

    A.fill( 2.f );
    B.fill( 2.f );
    Comp.fill( 4.f );

    Res = nto::elemul( A, B );

    std::cout << "elemul test : " << ( ( Res == Comp ) ? "PASSED" : "FAILED" ) << std::endl;

    // SUBSTRACT

    Comp.fill( 0.f );

    Res = A - B;

    std::cout << "substraction test : " << ( ( Res == Comp ) ? "PASSED" : "FAILED" ) << std::endl;

    // SIG

    Res.fill( 2.f );

    Comp.fill( 1.f / ( 1.f + std::exp(-2.f) ) );

    nto::sig( Res );

    //std::cout << Res.dump(0,0) << std::endl;
    //std::cout << Comp.dump(0,0) << std::endl;

    std::cout << "sig test : " << ( ( Res == Comp ) ? "PASSED" : "FAILED" ) << std::endl;

    // D_SIG

    A.fill( 2.f );
    Comp.fill( -2.f );

    Res = nto::d_sig( A );

    std::cout << "d_sig test : " << ( ( Res == Comp ) ? "PASSED" : "FAILED" ) << std::endl;

    // INCREMENT

    A.fill( 1.f );
    Res.fill( 1.f );
    Comp.fill( 2.f );

    Res += A;

    std::cout << "increment test : " << ( ( Res == Comp ) ? "PASSED" : "FAILED" ) << std::endl;

    // DECREMENT

    Res.fill( 1.f );
    Comp.fill( 0.f );

    Res -= A;

    std::cout << "decrement test : " << ( ( Res == Comp ) ? "PASSED" : "FAILED" ) << std::endl;

    // MULTRANS1

    A.resize(1,4,1,1);
    B.resize(1,4,1,1);

    A.fill( 2.f );
    B.fill( 2.f );
    Comp.fill( 4.f );

    Res = nto::multrans1( A, B );

    std::cout << "multrans1 test : " << ( ( Res == Comp ) ? "PASSED" : "FAILED" ) << std::endl;

    // MULTRANS2

    A.resize(4,1,1,1);
    B.resize(4,1,1,1);

    A.fill( 2.f );
    B.fill( 2.f );
    Comp.fill( 4.f );

    Res = nto::multrans2( A, B );

    std::cout << "multrans2 test : " << ( ( Res == Comp ) ? "PASSED" : "FAILED" ) << std::endl;

    return 0;
}
