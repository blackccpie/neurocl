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

#include "alpr.h"
#include "alphanum.h"

#include <boost/lexical_cast.hpp>

namespace alpr {

using namespace cimg_library;

// Network cell size
const size_t g_sizeX = 50;
const size_t g_sizeY = 100;

license_plate::license_plate( const std::string& file_plate, neurocl::network_manager& net_manager )
    : m_detector_output( new float[36] ), m_net_manager( net_manager )
{
    CImg<float> input_plate( file_plate.c_str() );

    _prepare_work_plate( input_plate );
}

license_plate::~license_plate()
{
}

void license_plate::_prepare_work_plate( CImg<float>& input_plate )
{
    // Compute reduced plate image
    m_work_plate = input_plate.resize( g_sizeY * input_plate.width() / input_plate.height(), g_sizeY );

    m_work_plate.channel(0);                // B&W (check position...binarization is different if called after inversion)
    m_work_plate.equalize( 256, 0, 255 );   // spread lut
    m_work_plate.normalize( 0.f, 1.f );     // normalize
    m_work_plate = 1.f - m_work_plate;      // invert

    m_work_plate.display();

    m_work_plate.threshold( 0.6f );

    // Remove a 10px border
    cimg_for_borderXY( m_work_plate, x, y, 10 ) { m_work_plate( x, y ) = 0; }

    m_work_plate.display();
}

void license_plate::_compute_ranges()
{
    // Compute row sums image
    CImg<float> row_sums( m_work_plate.width(), 1 );
    cimg_forX( row_sums, x )
    {
        float rsum = 0.f;
        cimg_forY( m_work_plate, y )
        {
            rsum += m_work_plate(x,y);
        }
        row_sums(x) = rsum;
    }

    // Display row sums graph
    CImg<float> sums_graph( m_work_plate.width(), 400, 1, 3, 0 );
    unsigned char green[] = { 0,255,0 };
    sums_graph.draw_graph( row_sums, green, 1, 1, 1 );//, 0, 100 );
    sums_graph.display();
}

void license_plate::_compute_distance_map()
{
    // Initialize distance map
    CImg<float> dist_map( m_work_plate.width(), 1, 1, 1, 0 );

    for ( size_t i=0; i<=( m_work_plate.width() - g_sizeX ); i++ )
    {
        CImg<float> subimage = m_work_plate.get_columns( i, i + g_sizeX - 1 );
        cimg_for_borderXY( subimage, x, y, 2 ) { subimage( x, y ) = 0; }
        //subimage.threshold( 0.5f );

        neurocl::sample sample( g_sizeX * g_sizeY, subimage.data(), 36, m_detector_output.get() );

        m_net_manager.compute_output( sample );

        //std::cout << sample.output() << std::endl;

        if ( sample.max_comp_val() > 0.8f )
        {
            dist_map(i) = sample.max_comp_val();

            if ( false )
            {
                CImg<float> disp_image( 50, 100, 1, 3 );
                cimg_forXYC( disp_image, x, y, c ) {
                    disp_image( x, y, c ) = 255.f * subimage( x, y ); }

                unsigned char green[] = { 0,255,0 };
                std::string label = alphanum( sample.max_comp_idx() ).string() + " "
                    + boost::lexical_cast<std::string>( sample.max_comp_val() );
                disp_image.draw_text( 5, 5, label.c_str(), green );
                disp_image.display();
            }
        }

        //std::cout << "TEST OUTPUT IS : " << sample.output() << std::endl;
    }

    // Display distance map graph
    CImg<float> dist_graph( m_work_plate.width(), 400, 1, 3, 0 );
    unsigned char red[] = { 255,0,0 };
    dist_graph.draw_graph( dist_map, red, 1, 1, 1 );//, 0, 100 );
    dist_graph.display();
}

void license_plate::analyze()
{
    _compute_ranges();
    _compute_distance_map();
}

}; //namespace alpr
