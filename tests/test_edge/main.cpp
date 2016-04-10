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

#include <boost/chrono.hpp>

int main( int argc, char *argv[] )
{
    CImg<float> image_in( "/Users/albertmurienne/Public/facecam_faces/E/0.png" );

    boost::chrono::system_clock::time_point start = boost::chrono::system_clock::now();

    image_in.resize( 50, 50 );
    image_in.equalize( 256, 0, 255 );
    image_in.normalize( 0.f, 1.f );
    image_in.channel(0);

    CImg<float> image_out( image_in.width(), image_in.height(), 1, 1 );

    sobel::process( image_in, image_out );

    boost::chrono::microseconds duration = boost::chrono::duration_cast<boost::chrono::microseconds>( boost::chrono::system_clock::now() - start );

    std::cout << "EDGE DETECTION IN : " << duration.count() << "us" << std::endl;

    image_out.display();

    return 0;
}
