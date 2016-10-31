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

#include "facetools/edge_detect.h"

#include <chrono>
#include <iostream>

//#define USE_CCV_IMP

template<typename T>
void run_test( const char* in )
{
    CImg<T> image_in( in );

    image_in.equalize( 256, 0, 255 );
#ifndef USE_CCV_IMP
    image_in.normalize( 0.f, 1.f );
#endif
    image_in.channel(0);

    CImg<T> image_out1( image_in.width(), image_in.height(), 1, 1 );
    CImg<T> image_out2( image_in.width(), image_in.height(), 1, 1 );

    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

#ifdef USE_CCV_IMP
    sobel_ccv::process<T>( image_in, image_out1 );
#else
    sobel::process<T>( image_in, image_out1 );
#endif

    std::chrono::microseconds duration = std::chrono::duration_cast<std::chrono::microseconds>( std::chrono::system_clock::now() - start );

    std::cout << "SOBEL EDGE DETECTION IN : " << duration.count() << "us" << std::endl;

    start = std::chrono::system_clock::now();

#ifdef USE_CCV_IMP
    canny_ccv::process<T>( image_in, image_out2 );
#else
    canny<T> can( image_in.width(), image_in.height() );
    can.process( image_in, image_out2 );
#endif

    duration = std::chrono::duration_cast<std::chrono::microseconds>( std::chrono::system_clock::now() - start );

    std::cout << "CANNY EDGE DETECTION IN : " << duration.count() << "us" << std::endl;

    CImgList<T> list( image_out1, image_out2 );

    list.display();
}

int main( int argc, char *argv[] )
{
#ifdef USE_CCV_IMP
    run_test<unsigned char>( argv[1] );
#else
    run_test<float>( argv[1] );
#endif

    return 0;
}
