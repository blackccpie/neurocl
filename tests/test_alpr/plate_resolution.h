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

#ifndef PLATE_RESOLUTION_H
#define PLATE_RESOLUTION_H

#include "network_manager.h"

#include "CImg.h"

namespace alpr {

// number of segments to identify
const size_t g_number_segments = 9;

// maximum number of tries per segment
const size_t g_max_try_per_segment = 10;

class plate_resolution
{
public:

    typedef enum
    {
        ANALYZING = 0,
        ANALYZE_NEXT,
        ANALYZE_ENDED,
        UNKNOWN
    } resolution_status;

public:
    plate_resolution( neurocl::network_manager& net_num, neurocl::network_manager& net_let );

    // Push a new candidate image and give the working segment index
    const resolution_status push_candidate( cimg_library::CImg<float>& candidate, const size_t segment_pos );

    // Compute results
    void compute_results();

    // Get last sample
    const boost::shared_ptr<neurocl::sample>& last_sample() { return m_sample; }

private:

    struct segment_status
    {
        segment_status() : retries(1) {}
        size_t retries;
        std::pair<size_t,float> order_scores[g_max_try_per_segment];
    };

private:

    void _preprocess_candidate( cimg_library::CImg<float>& candidate, bool separator );

private:

    segment_status m_segment_status[g_number_segments];

    boost::shared_array<float> m_num_output;
    boost::shared_array<float> m_let_output;

    boost::shared_ptr<neurocl::sample> m_sample;
    neurocl::network_manager& m_net_num;
    neurocl::network_manager& m_net_let;
};

}; //namespace alpr

#endif //PLATE_RESOLUTION_H
