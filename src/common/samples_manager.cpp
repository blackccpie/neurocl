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

#include "samples_manager.h"
#include "network_exception.h"
#include "logger.h"

#include "CImg.h"

#include <boost/filesystem.hpp>
namespace bfs = boost::filesystem;

#include <iostream>
#include <fstream>
#include <sstream>

namespace neurocl {

// Pad input image with zeros
void _pad( const cimg_library::CImg<float>& source, cimg_library::CImg<float>& padded, const size_t& pad_size )
{
    // NOTE : could have used CImg resize function

    padded.resize( source.width() + 2*pad_size, source.height() + 2*pad_size );

    cimg_for_borderXY( padded, x, y, pad_size ) { padded( x, y ) = 0; }
    cimg_for_insideXY( padded, x, y, pad_size ) { padded( x, y ) = source( x - pad_size, y - pad_size ); }
}

cimg_library::CImg<float> _get_preprocessed_image( const std::string& file )
{
    cimg_library::CImg<float> img( file.c_str() );

    img.normalize( 0.f, 1.f );
    img.channel(0);

    return img;
}

void samples_manager::load_samples( const std::string &input_filename, bool shuffle, t_preproc extra_preproc )
{
    m_sample_sizeX = m_sample_sizeY = 0;

    if ( !bfs::exists( input_filename ) )
    {
        LOGGER(error) << "samples_manager::load_samples - error reading input samples config file \'" << input_filename << "\'" << std::endl;
        throw network_exception( "error reading input samples config file" );
    }

    std::ifstream data_in( input_filename );
    if ( !data_in || !data_in.is_open() )
    {
        LOGGER(error) << "samples_manager::load_samples - error opening input samples config file \'" << input_filename << "\'" << std::endl;
        throw network_exception( "error reading input samples config file" );
    }

    // clear previous samples
    m_input_samples.clear();
    m_output_samples.clear();
    m_samples_set.clear();

    std::string line;

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
        cimg_library::CImg<float> img = _get_preprocessed_image( image_filename );

        if ( !m_sample_sizeX && !m_sample_sizeY )
        {
            m_sample_sizeX = img.width();
            m_sample_sizeY = img.height();

            m_augmenter = std::make_shared<samples_augmenter>( m_sample_sizeX, m_sample_sizeY );
        }
        else
        {
            if ( ( m_sample_sizeX != img.width() ) ||
                ( m_sample_sizeY != img.height() ) )
                throw network_exception( "non uniform sample size in input sample set" );
        }

        // manage custom preprocessing if needed
        if ( extra_preproc )
        {
            extra_preproc( img.data(), img.width(), img.height() );
        }

        // store input sample in list
        size_t input_size = img.size();
        boost::shared_array<float> input_sample( new float[input_size] );
        std::copy( img.data(), img.data()+input_size, input_sample.get() );
        m_input_samples.push_back( input_sample );

        // If they exist, read the target values from the rest of the line:
        std::vector<float> _vals;
        while ( !ss.eof() )
        {
            float val;
            if ( !(ss >> val).fail() )
                _vals.push_back( val );
        }

        // save output image
        size_t output_size = _vals.size();
        boost::shared_array<float> output_sample{ new float[output_size] };
        std::copy( _vals.begin(), _vals.end(), output_sample.get() );
        m_output_samples.push_back( output_sample );

        // store new sample
        m_samples_set.push_back( neurocl::sample( input_size, m_input_samples.back().get(), output_size, m_output_samples.back().get() ) );

        // manage restricted size
        if ( m_restrict_size == m_samples_set.size() )
            break;
    }

    if ( shuffle )
        std::random_shuffle( m_samples_set.begin(), m_samples_set.end() );
}

const std::vector<neurocl::sample> samples_manager::get_next_batch( const size_t size ) const noexcept
{
    if ( m_end )
        return std::vector<neurocl::sample>();

    auto begin = m_samples_set.cbegin() + m_batch_index;
    auto end = begin + size;

    if ( end >= m_samples_set.cend() )
    {
        end = m_samples_set.cend();
        m_end = true;
    }
    else
        m_batch_index += size;

    return std::vector<neurocl::sample>( begin, end );
}

void samples_manager::rewind() const noexcept
{
    m_end = false;
    m_batch_index = 0;
}

void samples_manager::shuffle() const noexcept
{
    std::random_shuffle( m_samples_set.begin(), m_samples_set.end() );
}

void samples_manager::_assert_sample_size() const
{
	if ( !m_sample_sizeX || !m_sample_sizeY )
        throw network_exception( "no sample set loaded yet (undefined sample 2D size)" );
}

std::shared_ptr<samples_augmenter> samples_manager::get_augmenter() const
{
    _assert_sample_size();

    return m_augmenter;
}

cimg_library::CImg<float> g_buf_img{};

samples_augmenter::samples_augmenter( const int sizeX, const int sizeY ) : m_sizeX( sizeX ), m_sizeY( sizeY )
{
}

neurocl::sample samples_augmenter::noise( const neurocl::sample& s, const float sigma ) const
{
	g_buf_img.assign( s.isample, m_sizeX, m_sizeY, 1, 1, false );
	g_buf_img.noise( sigma, 0 ); // gaussian

	return neurocl::sample( m_sizeX * m_sizeY, g_buf_img.data(), s.osample_size, s.osample );
}

neurocl::sample samples_augmenter::rotate( const neurocl::sample& s, const float angle ) const
{
	g_buf_img.assign( s.isample, m_sizeX, m_sizeY, 1, 1, false );
	g_buf_img.rotate( angle );
	g_buf_img.resize( m_sizeX, m_sizeY ); // rotate does not guarantee size conservation, cf. CImg documentation

	return neurocl::sample( m_sizeX * m_sizeY, g_buf_img.data(), s.osample_size, s.osample );
}

neurocl::sample samples_augmenter::translate( const neurocl::sample& s, const int sx, const int sy ) const
{
    if ( ( sx > 2 ) || ( sy > 2 ) )
        throw network_exception( "no translation over 2px manageed yet" );

	g_buf_img.assign( s.isample, m_sizeX, m_sizeY, 1, 1, false );
    g_buf_img.resize( m_sizeX+4, m_sizeY+4, -100, -100, 0, 0, 0.5f, 0.5f );
    int startX = 2 + sx;
    int startY = 2 + sy;
    g_buf_img.crop( startX, startY, startX + m_sizeX, startY + m_sizeY );

	return neurocl::sample( m_sizeX * m_sizeY, g_buf_img.data(), s.osample_size, s.osample );
}

}; //namespace neurocl
