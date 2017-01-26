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

        CImg<float> cropped_numbers = get_cropped_numbers( input );

        cropped_numbers.normalize( 0, 255 );
        auto_threshold( cropped_numbers );

        cropped_numbers.display();

        std::vector<t_number_interval> number_intervals;
        compute_ranges( cropped_numbers, number_intervals );

        float output[10] = { 0.f };

        CImg<float> cropped_numbers_res( cropped_numbers.width(), cropped_numbers.height(), 1, 3, 0 );
        cimg_forXY( cropped_numbers, x, y )
        {
            cropped_numbers_res( x, y, 0 ) = cropped_numbers_res( x, y, 1 ) = cropped_numbers_res( x, y, 2 ) = cropped_numbers( x, y );
        }
        cropped_numbers_res.normalize( 0, 255 );

        std::shared_ptr<samples_augmenter> smp_augmenter = std::make_shared<samples_augmenter>( 28, 28 );

        for ( auto& ni : number_intervals )
        {
            CImg<float> cropped_number( cropped_numbers.get_columns( ni.first, ni.second ) );

            center_number( cropped_number );

            sample sample( cropped_number.width() * cropped_number.height(), cropped_number.data(), 10, output );
            net_manager->compute_augmented_output( sample, smp_augmenter );

            std::cout << "max comp idx: " << sample.max_comp_idx() << " max comp val: " << sample.max_comp_val() << std::endl;

            //cropped_number.display();

            std::string item = std::to_string( sample.max_comp_idx() );
            std::string item_confidence = std::to_string( (int)( 100 * sample.max_comp_val() ) ) + "%%";
            cropped_numbers_res.draw_text( ni.first, 10, item.c_str(), green, 0, 1.f, 50 );
            cropped_numbers_res.draw_text( ni.first, 70, item_confidence.c_str(), red, 0, 1.f, 15 );
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
