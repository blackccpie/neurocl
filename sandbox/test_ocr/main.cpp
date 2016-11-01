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

#include "neurocl.h"

#include "../../apps/alpr/autothreshold.h"
#include "../../utils/facetools/edge_detect.h"

#include "CImg.h"

#include <iostream>

using namespace neurocl;
using namespace cimg_library;

CImg<> get_row_sums( const CImg<>& input )
{
    // Compute row sums image
    CImg<float> row_sums( input.width(), 1 );
    cimg_forX( row_sums, x )
    {
        auto rsum = 0.f;
        cimg_for_insideY( input, y, 0 ) // 0px margin
        {
            rsum += input(x,y);
        }
        row_sums(x) = rsum;
    }
    return row_sums;
}

CImg<> get_line_sums( const CImg<>& input )
{
    // Compute line sums image
    CImg<float> line_sums( 1, input.height() );
    cimg_forY( line_sums, y )
    {
        auto lsum = 0.f;
        cimg_for_insideX( input, x, 0 ) // 0px margin
        {
            lsum += input(x,y);
        }
        line_sums(y) = lsum;
    }
    return line_sums;
}

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
        //std::shared_ptr<network_manager_interface> net_manager = network_factory::build();
        //net_manager->load_network( "../nets/mnist/topology-mnist2.txt", "../nets/mnist/weights-mnist2.bin" );

    	CImg<unsigned char> input( argv[1] );

        input.channel(0);

        CImg<unsigned char> input_edge( input.width(), input.height(), 1, 1 );

        sobel_ccv::process<unsigned char>( input, input_edge );

        input_edge.normalize( 0, 255 );
        input_edge.dilate( 2 );
        input_edge.threshold( 40 );
        input_edge.display();

        // Compute row sums image
        CImg<unsigned char> row_sums = get_row_sums( input_edge );
        row_sums.threshold( 5 );
        row_sums.display();

        // Compute line sums image
        CImg<unsigned char> line_sums = get_line_sums( input_edge );
        line_sums.threshold( 5 );
        line_sums.display();

        // Compute extraction coords
        int startX = 0;
        int stopX = 0;
        cimg_forX( row_sums, x )
        {
            if ( row_sums(x) )
            {
                startX = x;
                break;
            }
        }
        cimg_forX( row_sums, x )
        {
            if ( row_sums( row_sums.width() - x - 1 ) )
            {
                stopX = row_sums.width() - x - 1;
                break;
            }
        }

        int startY = 0;
        int stopY = 0;
        cimg_forY( line_sums, y )
        {
            if ( line_sums(y) )
            {
                startY = y;
                break;
            }
        }
        cimg_forY( line_sums, y )
        {
            if ( line_sums( line_sums.height() - y - 1 ) )
            {
                stopY = line_sums.height() - y - 1;
                break;
            }
        }

        int margin = ( stopY - startY ) / 7; // empirical ratio...
        startX -= margin;
        startY -= margin;
        stopX += margin;
        stopY += margin;

        std::cout << margin << " / " << startX << " " << startY << " " << stopX << " " << stopY << std::endl;

        CImg<unsigned char> cropped = input.get_crop( startX, startY, stopX, stopY );
        cropped.display();
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
