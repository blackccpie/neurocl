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

#include "plate_resolution.h"
#include "alphanum.h"

#include <boost/lexical_cast.hpp>
#include <boost/make_shared.hpp>

#include <iostream>
#include <vector>

namespace alpr {

#define DISPLAY_CANDIDATES

const std::vector<size_t> french_plate_numbers_pos = list_of (4)(5)(6);
const std::vector<size_t> french_plate_letters_pos = list_of (1)(2)(8)(9);
const std::vector<size_t> french_plate_separators_pos = list_of (3)(7);

bool is_number_pos( const size_t pos )
{ return ( std::find( french_plate_numbers_pos.begin(), french_plate_numbers_pos.end(), pos ) != french_plate_numbers_pos.end() ); }

bool is_letter_pos( const size_t pos )
{ return ( std::find( french_plate_letters_pos.begin(), french_plate_letters_pos.end(), pos ) != french_plate_letters_pos.end() ); }

bool is_separator_pos( const size_t pos )
{ return ( std::find( french_plate_separators_pos.begin(), french_plate_separators_pos.end(), pos ) != french_plate_separators_pos.end() ); }

plate_resolution::plate_resolution( neurocl::network_manager& net_num, neurocl::network_manager& net_let )
    :   m_net_num( net_num ), m_net_let( net_let ),
        m_num_output( new float[10] ), m_let_output( new float[27] )
{
}

void plate_resolution::_preprocess_candidate( cimg_library::CImg<float>& candidate, bool separator )
{
    // remove border
    //cimg_for_borderXY( candidate, x, y, 3 ) { candidate( x, y ) = 0; } // TODO-AM : configurable?

    // if separator clear image outside "ROI"
    if ( separator )
    {
        //**** TEMPORARY HACK ****//
        cimg_for_borderY( candidate, y, 30 )
            cimg_forX( candidate, x )
                candidate( x, y ) = 0;
        cimg_forY( candidate, y )
            for ( int x = ( candidate.width() - 16 ); x<candidate.width(); x++ )
                candidate( x, y ) = 0;
    }
}

const plate_resolution::resolution_status plate_resolution::push_candidate( cimg_library::CImg<float>& candidate, const size_t segment_pos )
{
    bool separator = false;
    alphanum::data_type type = alphanum::UNKNOWN;

    if ( is_letter_pos( segment_pos ) )
    {
        type = alphanum::LETTER;
    }
    else if ( is_number_pos( segment_pos ) )
    {
        type = alphanum::NUMBER;
    }
    else if ( is_separator_pos( segment_pos ) )
    {
        type = alphanum::LETTER;
        separator = true;
    }
    else
    {
        std::cout << "WARNING -> abnormal item position " << segment_pos << std::endl;
        return ANALYZE_ENDED;
    }

    // if max retries reached, switch to next segment
    size_t& cur_retries = m_segment_status[segment_pos].retries;
    if ( cur_retries > g_max_try_per_segment )
        return ANALYZE_NEXT;

    std::cout << "SEGMENT " << segment_pos << "/" << cur_retries << std::endl;

    // pre-process candidate image
    _preprocess_candidate( candidate, separator );

    switch( type )
    {
    case alphanum::LETTER:
        m_sample = boost::make_shared<neurocl::sample>( candidate.width() * candidate.height(), candidate.data(), 27, m_let_output.get() );
        m_net_let.compute_output( *m_sample );
        break;
    case alphanum::NUMBER:
        m_sample = boost::make_shared<neurocl::sample>( candidate.width() * candidate.height(), candidate.data(), 10, m_num_output.get() );
        m_net_num.compute_output( *m_sample );
        break;
    case alphanum::UNKNOWN:
    default:
        // should never be reached
        break;
    }

#ifdef DISPLAY_CANDIDATES
    cimg_library::CImg<float> disp_image( 50, 100, 1, 3 );
    cimg_forXYC( disp_image, x, y, c ) {
        disp_image( x, y, c ) = 255.f * candidate( x, y ); }

    unsigned char green[] = { 0,255,0 };
    std::string label = alphanum( m_sample->max_comp_idx(), type ).string() + " "
        + boost::lexical_cast<std::string>( m_sample->max_comp_val() );
    disp_image.draw_text( 5, 5, label.c_str(), green );
    disp_image.display();
#endif

    ++cur_retries;

    return ANALYZING;
}

void compute_results()
{

}

}; //namespace alpr
