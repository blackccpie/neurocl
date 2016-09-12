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

#include "CImg.h"

#include <boost/filesystem.hpp>
namespace bfs = boost::filesystem;

#include <fstream>
#include <iostream>
#include <sstream>

namespace neurocl {

cimg_library::CImg<float> _get_preprocessed_image( const std::string& file )
{
    cimg_library::CImg<float> img( file.c_str() );

    img.resize(32,32);

    img.equalize( 256, 0, 255 );
    img.normalize( 0.f, 1.f );
    img.channel(0);

    return img;
}

void samples_manager::load_samples( const std::string &input_filename, bool shuffle, t_preproc extra_preproc )
{
    if ( !bfs::exists( input_filename ) )
    {
        std::cerr << "samples_manager::load_samples - error reading input samples config file \'" << input_filename << "\'" << std::endl;
        throw network_exception( "error reading input samples config file" );
    }

    std::ifstream data_in( input_filename );
    if ( !data_in || !data_in.is_open() )
    {
        std::cerr << "samples_manager::load_samples - error opening input samples config file \'" << input_filename << "\'" << std::endl;
        throw network_exception( "error reading input samples config file" );
    }

    // clear previous samples
    m_input_samples.clear();
    m_output_samples.clear();
    m_samples_set.clear();

    std::string line;

    while ( std::getline( data_in, line ) )
    {
        std::stringstream ss( line );

        // parse image filename
        std::string image_filename;
        ss >> image_filename;

        // skip blank and comment lines:
        if ( image_filename.size() == 0 || image_filename[0] == '#')
            continue;

        // preprocess and save input image
        cimg_library::CImg<float> img = _get_preprocessed_image( image_filename );

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
        boost::shared_array<float> output_sample( new float[output_size] );
        std::copy( _vals.begin(), _vals.end(), output_sample.get() );
        m_output_samples.push_back( output_sample );

        // store new sample
        m_samples_set.push_back( neurocl::sample( input_size, m_input_samples.back().get(), output_size, m_output_samples.back().get() ) );
    }

    if ( shuffle )
        std::random_shuffle( m_samples_set.begin(), m_samples_set.end() );
}

const std::vector<neurocl::sample> samples_manager::get_next_batch( const size_t size ) const
{
    if ( m_end )
        return std::vector<neurocl::sample>();

    std::vector<neurocl::sample>::const_iterator begin = m_samples_set.begin() + m_batch_index;
    std::vector<neurocl::sample>::const_iterator end = begin + size;

    if ( end >= m_samples_set.end() )
    {
        end = m_samples_set.end();
        m_end = true;
    }
    else
        m_batch_index += size;

    return std::vector<neurocl::sample>( begin, end );
}

void samples_manager::rewind() const
{
    m_end = false;
    m_batch_index = 0;
}

}; //namespace neurocl
