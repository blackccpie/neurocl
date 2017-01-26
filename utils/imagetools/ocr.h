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

#include "autothreshold.h"
#include "edge_detect.h"

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

CImg<float> get_cropped_numbers( const CImg<float>& input )
{
    CImg<unsigned char> work( input );
    CImg<unsigned char> work_edge( work );

    sobel_ccv::process<unsigned char>( work, work_edge );

    work_edge.normalize( 0, 255 );
    //work_edge.dilate( 2 );
    work_edge.threshold( 40 );
    //input_edge.display();

    // Compute row sums image
    CImg<unsigned char> row_sums = get_row_sums( work_edge );
    row_sums.threshold( 5 );
    //row_sums.display();

    // Compute line sums image
    CImg<unsigned char> line_sums = get_line_sums( work_edge );
    line_sums.threshold( 5 );
    //line_sums.display();

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
    startX -= 2 * margin;
    startY -= margin;
    stopX += 2 * margin;
    stopY += margin;

    //std::cout << margin << " / " << startX << " " << startY << " " << stopX << " " << stopY << std::endl;

    CImg<float> cropped( input.get_crop( startX, startY, stopX, stopY ) );
    cropped = 1.f - cropped;

    return cropped;
}

using t_number_interval = std::pair<size_t,size_t>;
void compute_ranges( const CImg<float>& input, std::vector<t_number_interval>& number_intervals )
{
    // Compute row sums image
    CImg<float> row_sums = get_row_sums( input );
    row_sums.threshold( 2.f );

    //row_sums.display();

    // Detect letter ranges
    size_t first = 0;
    bool last_val = false;
    cimg_forX( row_sums, x )
    {
        bool cur_val = row_sums[x] > 0.f;

        if ( cur_val != last_val )
        {
            if ( !last_val ) // ascending front
            {
                first = x;
            }
            else //descending front
            {
                if ( first != 0 )
                {
                    std::cout << "detected interval " << first << " " << x << std::endl;
                    number_intervals.push_back( std::make_pair( first, x ) );
                    first = 0;
                }
            }
        }
        last_val = cur_val;
    }
}

void center_number( CImg<float>& input )
{
    // Compute row sums image
    CImg<float> row_sums = get_row_sums( input );
    row_sums.threshold( 2.f );
    //row_sums.display();

    int startX, stopX;
    bool last_val = false;
    cimg_forX( row_sums, x )
    {
        bool cur_val = row_sums[x] > 0.f;

        if ( cur_val != last_val )
        {
            if ( !last_val ) // ascending front
                startX = x;
            else //descending front
            {
                stopX = x;
                break;
            }
        }
        last_val = cur_val;
    }

    // Compute line sums image
    CImg<float> line_sums = get_line_sums( input );
    line_sums.threshold( 2.f );
    //line_sums.display();

    int startY, stopY;
    last_val = false;
    cimg_forY( line_sums, y )
    {
        bool cur_val = line_sums[y] > 0.f;

        if ( cur_val != last_val )
        {
            if ( !last_val ) // ascending front
                startY = y;
            else //descending front
            {
                stopY = y;
                break;
            }
        }
        last_val = cur_val;
    }

    // try to prepare image like MNIST does:
    // http://yann.lecun.com/exdb/mnist/

    //int max_dim = std::max( input.width(), input.height() );
    int max_dim = std::max( stopX - startX, stopY - startY );

    input.crop( startX, startY, stopX, stopY);
    input.resize( max_dim, max_dim, -100, -100, 0, 0, 0.5f, 0.5f );
    input.resize( 20, 20, -100, -100, 6 );
    input.normalize( 0, 255 );

    // compute center of mass
    int massX = 0;
    int massY = 0;
    int num = 0;
    cimg_forXY( input, x, y )
    {
        massX += input( x, y ) * x;
        massY += input( x, y ) * y;
        num += input( x, y );
    }
    massX /= num;
    massY /= num;

    std::cout << "Mass center X=" << massX << " Y=" << massY << std::endl;

    input.resize( 28, 28, -100, -100, 0, 0, 1.f - ((float)massX)/20.f, 1.f - ((float)massY)/20.f );

    input.normalize( 0.f, 1.f );

    //input.display();
}
