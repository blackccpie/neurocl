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

#include "network_manager.h"
#include "network_exception.h"

#include "CImg.h"

#include <boost/lexical_cast.hpp>

#include <iostream>

using namespace cimg_library;

#define IMAGE_SIZEX 480
#define IMAGE_SIZEY 320

#define NEUROCL_EPOCH_SIZE 30
#define NEUROCL_BATCH_SIZE 20
#define MAX_MATCH_ERROR 0.1f

int main( int argc, char *argv[] )
{
    std::cout << "Welcome to test_denoising!" << std::endl;

    if ( argc == 1 )
    {
        std::cout << "Invalid arguments!" << std::endl;
        std::cout << "example: ./test_denoising denoise-train.txt topology-denoise.txt weights-denoise.bin" << std::endl;
        return -1;
    }

    try
    {
        bool denoise1_or_train0 = ( boost::lexical_cast<int>( argv[1] ) == 1 );

        neurocl::network_manager net_manager( neurocl::network_manager::NEURAL_IMPL_BNU_REF );
        net_manager.load_network( argv[2], argv[3] );

    	CImg<unsigned char> input_image( IMAGE_SIZEX, IMAGE_SIZEY, 1, 3 );

        // Train on 1000 images
    	for ( auto i=0; i<1000; i++ )
    	{
        }
    }
    catch( neurocl::network_exception& e )
    {
        std::cerr << "network exception : " << e.what() << std::endl;
    }
    catch( std::exception& e )
    {
        std::cerr << "std::exception : " << e.what() << std::endl;
    }
    catch(...)
    {
        std::cerr << "unknown exception" << std::endl;
    }

    std::cout << "Bye bye test_denoising!" << std::endl;

    return 0;
}
