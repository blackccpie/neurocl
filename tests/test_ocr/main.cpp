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

#include "neurocl.h"

#include "imagetools/ocr.h"

unsigned char green[] = { 0,255,0 };
unsigned char red[] = { 255,0,0 };

using namespace neurocl;

int main( int argc, char *argv[] )
{
    std::cout << "Welcome to test_ocr!" << std::endl;

    if ( argc == 1 )
    {
        std::cout << "Invalid arguments!" << std::endl;
        std::cout << "example: ./test_ocr input.png" << std::endl;
        return -1;
    }

    try
    {
        std::shared_ptr<network_manager_interface> net_manager = network_factory::build();
        net_manager->load_network( "../nets/mnist/topology-mnist-kaggle.txt", "../nets/mnist/weights-mnist-kaggle.bin" );

    	CImg<unsigned char> input( argv[1] );
        input.channel(0);

        ocr_helper helper( net_manager );
        helper.process( input );

        CImg<float> cropped_numbers = helper.cropped_numbers();

        CImg<float> cropped_numbers_res( cropped_numbers.width(), cropped_numbers.height(), 1, 3, 0 );
        cimg_forXY( cropped_numbers, x, y )
        {
            cropped_numbers_res( x, y, 0 ) = cropped_numbers_res( x, y, 1 ) = cropped_numbers_res( x, y, 2 ) = cropped_numbers( x, y );
        }
        cropped_numbers_res.normalize( 0, 255 );

        for ( auto& reco : helper.recognitions() )
        {
            std::string item = std::to_string( reco.value );
            std::string item_confidence = std::to_string( (int)( reco.confidence ) ) + "%%";
            cropped_numbers_res.draw_text( reco.position, 10, item.c_str(), green, 0, 1.f, 50 );
            cropped_numbers_res.draw_text( reco.position, 70, item_confidence.c_str(), red, 0, 1.f, 15 );
        }

        cropped_numbers_res.display();
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

    std::cout << "Bye bye test_ocr!" << std::endl;

    return 0;
}
