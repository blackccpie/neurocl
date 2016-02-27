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

const std::string plate_resolution::segment_status::identified_segment() const
{
    switch( type )
    {
        case alphanum::LETTER:
            return v_letters_order[ max_comp_idx() ];
        case alphanum::NUMBER:
            return v_numbers_order[ max_comp_idx() ];
        case alphanum::SEPARATOR:
            return v_separators_order[ max_comp_idx() ];
        default:
            return "unidentified";
    }
}

plate_resolution::plate_resolution( neurocl::network_manager& net_num, neurocl::network_manager& net_let )
    :   m_net_num( net_num ), m_net_let( net_let ),
        m_num_output( 10 ), m_let_output( 26 )
{
    _build_segments();
}

void plate_resolution::_build_segments()
{
    m_segment_status.push_back( segment_status( alphanum::LETTER, 26 ) );
    m_segment_status.push_back( segment_status( alphanum::LETTER, 26 ) );
    m_segment_status.push_back( segment_status( alphanum::SEPARATOR, 1 ) );
    m_segment_status.push_back( segment_status( alphanum::NUMBER, 10 ) );
    m_segment_status.push_back( segment_status( alphanum::NUMBER, 10 ) );
    m_segment_status.push_back( segment_status( alphanum::NUMBER, 10 ) );
    m_segment_status.push_back( segment_status( alphanum::SEPARATOR, 1 ) );
    m_segment_status.push_back( segment_status( alphanum::LETTER, 26 ) );
    m_segment_status.push_back( segment_status( alphanum::LETTER, 26 ) );
}

void plate_resolution::_preprocess_candidate( cimg_library::CImg<float>& candidate, bool separator )
{
    // remove border
    cimg_for_borderXY( candidate, x, y, 3 ) { candidate( x, y ) = 0; } // TODO-AM : configurable?

    // if separator clear image outside "ROI"
    /*if ( separator )
    {
        ////// TEMPORARY HACK //////
        cimg_for_borderY( candidate, y, 30 )
            cimg_forX( candidate, x )
                candidate( x, y ) = 0;
        cimg_forY( candidate, y )
            for ( int x = ( candidate.width()/2 ); x<candidate.width(); x++ )
                candidate( x, y ) = 0;
    }
    // else remove eventually next caracter parts on the right of the image
    else
    {
        cimg_forY( candidate, y )
            for ( int x = ( 9*candidate.width()/10 ); x<candidate.width(); x++ )
                candidate( x, y ) = 0;
    }*/
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
        type = alphanum::SEPARATOR;
    }
    else
    {
        std::cout << "WARNING -> abnormal item position " << segment_pos << std::endl;
        return ANALYZE_ENDED;
    }

    // if max retries reached, switch to next segment
    size_t& cur_retries = m_segment_status[segment_pos-1].retries;
    if ( cur_retries > g_max_try_per_segment )
        return ANALYZE_NEXT;

    std::cout << "SEGMENT " << segment_pos << "/" << cur_retries << std::endl;

    // pre-process candidate image
    _preprocess_candidate( candidate, false/*separator*/ ); // TODO : separator management usefull?

    size_t candidate_max_comp_idx = 0;
    float candidate_max_comp_val = 0.f;

    switch( type )
    {
    case alphanum::LETTER:
        m_sample = boost::make_shared<neurocl::sample>( candidate.width() * candidate.height(), candidate.data(), 26, &m_let_output.data()[0] );
        m_net_let.compute_output( *m_sample );
        candidate_max_comp_idx = m_sample->max_comp_idx();
        candidate_max_comp_val = m_sample->max_comp_val();
        m_segment_status[segment_pos-1].accumulated_scores += m_let_output;
        break;
    case alphanum::NUMBER:
        m_sample = boost::make_shared<neurocl::sample>( candidate.width() * candidate.height(), candidate.data(), 10, &m_num_output.data()[0] );
        m_net_num.compute_output( *m_sample );
        candidate_max_comp_idx = m_sample->max_comp_idx();
        candidate_max_comp_val = m_sample->max_comp_val();
        m_segment_status[segment_pos-1].accumulated_scores += m_num_output;
        break;
    case alphanum::SEPARATOR:
        // TEMP-TODO -> compute a score given norm difference with separator image
        m_segment_status[segment_pos-1].accumulated_scores += boost::numeric::ublas::scalar_vector<float>( 1, 1.f );
        candidate_max_comp_idx = 0;
        candidate_max_comp_val = 1.f;
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
    std::string label = alphanum( candidate_max_comp_idx, type ).string() + " "
        + boost::lexical_cast<std::string>( candidate_max_comp_val );
    disp_image.draw_text( 5, 5, label.c_str(), green );
    disp_image.display();
#endif

    ++cur_retries;

    return ANALYZING;
}

const std::string plate_resolution::compute_results()
{
    std::string plate = _dump_plate();
    std::cout << "DETECTED PLATE IS " << plate << std::endl;
    return plate;
}

const std::string plate_resolution::_dump_plate()
{
    std::string plate_string = "";

    BOOST_FOREACH( const segment_status& status, m_segment_status )
    {
        //std::cout << status.max_comp_idx() << " " << status.max_comp_val() << std::endl;
        plate_string += status.identified_segment();
    }

    return plate_string;
}

}; //namespace alpr
