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

#include <sstream>
#include <string>

namespace neurocl {

struct sample
{
    sample( const size_t isize, const float* idata, const size_t osize, float* odata )
        : isample_size( isize ), isample( idata ), osample_size( osize ), osample( odata ) {}

    const std::string output()
    {
        std::stringstream ss;
        for ( size_t i=0; i<osample_size; i++ )
            ss << osample[i] << ";";
        return ss.str();
    }

    const size_t biggest_component()
    {
        return std::max_element( osample, osample + osample_size ) - osample;
    }

    size_t isample_size;
    const float* isample;
    size_t osample_size;
    float* osample; // TODO-AM : should be const for training samples!
};

} //namespace neurocl

#endif //NETWORK_SAMPLE_H
