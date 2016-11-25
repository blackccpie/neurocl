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

#include "CImg.h"

#include <iostream>
#include <fstream>
#include <limits>
#include <sstream>
#include <vector>

using namespace cimg_library;

//#define MEAN_STDDEV_COMPUTED

int main( int argc, char *argv[] )
{
    std::cout << "Welcome to mnist_normalizer!" << std::endl;

    std::ifstream data_in( "../nets/mnist/training/mnist-validate.txt" );

#ifndef MEAN_STDDEV_COMPUTED
    cimg_library::CImg<float> mean_img( 28, 28, 1, 1, 0.f );
    cimg_library::CImg<float> stddev_img( 28, 28, 1, 1, 0.f );

    std::string line;

    size_t N = 0;

    std::vector<cimg_library::CImg<float>> v_images;

    while ( std::getline( data_in, line ) )
    {
        std::stringstream ss{ line };

        // parse image filename
        std::string image_filename;
        ss >> image_filename;

        // skip blank and comment lines:
        if ( image_filename.size() == 0 || image_filename[0] == '#')
            continue;

        // preprocess and save input image
        cimg_library::CImg<float> img( image_filename.c_str() );
        img.channel(0);

        std::cout << "pushed image " << N << std::endl;

        v_images.push_back( img );

        mean_img += img;
        ++N;

    }

    mean_img /= N;

    mean_img.save( "mnist-mean.png" );
#else
    cimg_library::CImg<float> mean_img( "mnist-mean.png" );
#endif

    mean_img.display();

#ifndef MEAN_STDDEV_COMPUTED
    N = 0;

    float min = std::numeric_limits<float>::max();
    float max = std::numeric_limits<float>::min();

    for ( const auto& img : v_images )
    {
        min = ( img.min() < min ) ? img.min() : min;
        max = ( img.max() > max ) ? img.max() : max;
        stddev_img += ( img - mean_img ).pow( 2 );
    }

    stddev_img /= N-1;
    stddev_img.sqrt();

    stddev_img.save( "mnist-stddev.png" );
#else
    cimg_library::CImg<float> stddev_img( "mnist-stddev.png" );
#endif

    stddev_img.display();

    // test on first image
    cimg_library::CImg<float> test_img( "../nets/mnist/training/validate-data/0.bmp" );
    test_img.channel(0);
    cimg_library::CImg<float> preproc_img = ( test_img - mean_img ).div( stddev_img );

    preproc_img.display();

    std::cout << "Bye bye mnist_normalizer!" << std::endl;

    return 0;
}
