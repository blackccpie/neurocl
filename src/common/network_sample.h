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

#ifndef NETWORK_SAMPLE_H
#define NETWORK_SAMPLE_H

#include <boost/math/special_functions/pow.hpp>
#include <boost/shared_array.hpp>

#include <sstream>
#include <string>

namespace neurocl {

// Structure to store input sample
struct sample
{
    sample( const size_t isize, const float* idata, const size_t osize, const float* odata )
        : isample_size( isize ), isample( idata ), osample_size( osize ), osample( odata ) {}

    std::string output()
    {
        std::stringstream ss;
        for ( size_t i=0; i<osample_size; i++ )
            ss << osample[i] << ";";
        return ss.str();
    }

    size_t max_comp_idx()
    {
        return std::distance( osample, std::max_element( osample, osample + osample_size ) );
    }

    float max_comp_val()
    {
        return *std::max_element( osample, osample + osample_size );
    }

    size_t isample_size;
    const float* isample;
    size_t osample_size;
    const float* osample;
};

// Structure to store input test sample
struct test_sample : sample
{
    test_sample( const sample& s )
        : sample( s.isample_size, s.isample, s.osample_size, s.osample )
    {
        osample_ref.reset( new float[osample_size] );
        std::copy( osample, osample + osample_size, osample_ref.get() );
    }

    float RMSE()
    {
        float sum = 0.f;
        for ( size_t i=0; i<osample_size; i++ )
            sum += boost::math::pow<2>( osample[i] - osample_ref[i] );

        return std::sqrt( sum / osample_size );
    }

    std::string ref_output()
    {
        std::stringstream ss;
        for ( size_t i=0; i<osample_size; i++ )
            ss << osample_ref[i] << ";";
        return ss.str();
    }

    bool classified()
    {
        size_t biggest = std::max_element( osample, osample + osample_size ) - osample;
        size_t biggest_ref = std::max_element( osample_ref.get(), osample_ref.get() + osample_size ) - osample_ref.get();

        return ( biggest == biggest_ref );
    }

    void restore_ref()
    {
        std::copy( osample_ref.get(), osample_ref.get() + osample_size, const_cast<float*>( osample ) );
    }

    boost::shared_array<float> osample_ref;  // reference output sample buffer
};

} //namespace neurocl

#endif //NETWORK_SAMPLE_H
