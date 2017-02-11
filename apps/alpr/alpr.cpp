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

//#define cimg_plugin1 "plugins/nlmeans.h"
//#define cimg_plugin2 "plugins/loop_macros.h"

#include "alpr.h"
#include "plate_resolution.h"

#include "imagetools/autothreshold.h"

#include <iostream>

namespace alpr {

using namespace cimg_library;

#define DISPLAY_ROW_SUMS
#define DISPLAY_DISTANCE_MAP

// Network cell size
constexpr size_t g_sizeX = 50;
constexpr size_t g_sizeY = 100;

// Letters allowed range : TODO : ratio of total width?
constexpr size_t g_insideX = 10;

license_plate::license_plate(   const std::string& file_plate,
                                std::shared_ptr<neurocl::network_manager_interface> net_num,
                                std::shared_ptr<neurocl::network_manager_interface> net_let  )
    : m_plate_resol( net_num, net_let )
{
    // Initialize & prepare input plate image
    m_input_plate.load( file_plate.c_str() );
    _prepare_work_plate();
}

license_plate::~license_plate()
{
}

void license_plate::_prepare_work_plate()
{
    // Compute reduced plate image
    m_work_plate = m_input_plate.resize( g_sizeY * m_input_plate.width() / m_input_plate.height(), g_sizeY );

    m_work_plate.channel(0);                // B&W (check position...binarization is different if called after inversion)
    //m_work_plate.blur_median( 1 );          // Remove some noise
    //m_work_plate.nlmeans();

    // "Smart" binarization
    _smart_threshold();

    m_work_plate.normalize( 0.f, 1.f );     // normalize
    m_work_plate = 1.f - m_work_plate;      // invert

    // Remove a 10px border
    cimg_for_borderXY( m_work_plate, x, y, 10 ) { m_work_plate( x, y ) = 0; }

    m_work_plate.display();
}

void license_plate::_smart_threshold()
{
    auto_threshold( m_work_plate.data(), m_work_plate.width(), m_work_plate.height() );
    //m_work_plate.erode(2);
    //m_work_plate.dilate(2);
}

void license_plate::_compute_ranges()
{
    // Compute row sums image
    CImg<float> row_sums( m_work_plate.width(), 1 );
    cimg_forX( row_sums, x )
    {
        auto rsum = 0.f;
        cimg_for_insideY( m_work_plate, y, 10 )
        {
            rsum += m_work_plate(x,y);
        }
        row_sums(x) = rsum;
    }
    row_sums = 1.f - row_sums.normalize( 0.f, 1.f );
    //row_sums.save( "row_sums.png" );
    row_sums.threshold( 0.9f );

    // Detect letter ranges
    size_t first = 0;
    bool last_val = false;
    cimg_for_insideX( row_sums, x, g_insideX )
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
                    m_letter_intervals.push_back( std::make_pair( first, x ) );
                    first = 0;
                }
            }
        }
        last_val = cur_val;
    }

#ifdef DISPLAY_ROW_SUMS
    // Display row sums graph
    CImg<float> sums_graph( m_work_plate.width(), 400, 1, 3, 0 );
    unsigned char green[] = { 0,255,0 };
    sums_graph.draw_graph( row_sums, green, 1, 1, 1 );
    sums_graph.display();
#endif
}

void license_plate::_compute_distance_map()
{
    // Initialize distance map
    CImg<float> dist_map( m_work_plate.width(), 1, 1, 1, 0 );

    auto range_iter = m_letter_intervals.cbegin();

    size_t item_count = 1;

    // scroll the plate from left to right
    for ( size_t i=0; i<=( m_work_plate.width() - g_sizeX ); i++ )
    {
        if ( i < range_iter->first )
            continue;

        if ( i == range_iter->second )
        {
            item_count++;
            range_iter++;
            continue;
        }

        CImg<float> subimage = m_work_plate.get_columns( i, i + g_sizeX - 1 );

        plate_resolution::resolution_status status = m_plate_resol.push_candidate( subimage, item_count );

        switch( status )
        {
        case plate_resolution::resolution_status::ANALYZING:
            break;
        case plate_resolution::resolution_status::ANALYZE_NEXT:
            item_count++; // go to next item
            range_iter++; // go to next range
            break;
        case plate_resolution::resolution_status::ANALYZE_ENDED:
            break;
        case plate_resolution::resolution_status::UNKNOWN:
        default:
            // should never happen!
            break;
        }

        const std::shared_ptr<neurocl::sample>& sample = m_plate_resol.last_sample();

        if ( sample->max_comp_val() > 0.5f )
        {
            dist_map(i) = sample->max_comp_val();
        }

        //std::cout << "TEST OUTPUT IS : " << sample.output() << std::endl;
    }

    unsigned char blue[] = { 0,0,255 };
    unsigned char green[] = { 0,255,0 };
    unsigned char red[] = { 255,0,0 };
    unsigned char black[] = { 0,0,0 };

#ifdef DISPLAY_DISTANCE_MAP
    // Display distance map graph
    CImg<float> dist_graph( m_work_plate.width(), 400, 1, 3, 0 );
    dist_graph.draw_graph( dist_map, red, 1, 1, 1 );
    dist_graph.display();
#endif

    const std::string plate = m_plate_resol.compute_results();

    m_input_plate.draw_rectangle( 0, 0, m_input_plate.width(), m_input_plate.height(), black, 0.5f );
    std::string global_confidence = std::to_string( (int)( 100 * m_plate_resol.global_confidence() ) ) + "%%";
    m_input_plate.draw_text( 10, 10, global_confidence.c_str(), blue, 0, 1.f, 30 );
    int idx = 0;
    for( const auto& interval : m_letter_intervals )
    {
        if ( idx >= plate.size() )
            continue;

        std::string item( 1, plate.at(idx) );
        std::string item_confidence = std::to_string( (int)( 100 * m_plate_resol.confidence( idx ) ) ) + "%%";
        m_input_plate.draw_text( interval.second, 10, item.c_str(), green, 0, 1.f, 50 );
        m_input_plate.draw_text( interval.second, 70, item_confidence.c_str(), red, 0, 1.f, 15 );
        idx++;
    }
    m_input_plate.display();
}

void license_plate::analyze()
{
    _compute_ranges();
    _compute_distance_map();
}

}; //namespace alpr
