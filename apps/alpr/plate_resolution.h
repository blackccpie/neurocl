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

#include "alphanum.h"
#include "network_manager.h"

#include "CImg.h"

#include <boost/numeric/ublas/vector.hpp>

typedef boost::numeric::ublas::vector<float> vectorF;

namespace alpr {

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

    // Compute results and get detected plate string
    const std::string compute_results();

    // Access confidence results
    const float global_confidence();
    const float confidence( const size_t idx );

    // Get last sample
    const boost::shared_ptr<neurocl::sample>& last_sample() { return m_sample; }

private:

    struct segment_status
    {
        segment_status( const alphanum::data_type& _type, const size_t& _scores_size )
            : retries(1), type( _type ), accumulated_scores( _scores_size ) { accumulated_scores.clear(); }
        segment_status( const segment_status& status )
            : retries( status.retries ), type( status.type ), accumulated_scores( status.accumulated_scores ) {}
        alphanum::data_type type;
        size_t retries;
        vectorF accumulated_scores;

        const float confidence() const
        {
            return norm_inf( accumulated_scores ) / norm_1( accumulated_scores );
        }

        const size_t max_comp_idx() const
        {
            return index_norm_inf( accumulated_scores );
        }

        const float max_comp_val() const
        {
            return norm_inf( accumulated_scores ); // maximum norm (biggest absolute real value)
        }

        const std::string identified_segment() const;
    };

private:

    void _build_segments();
    void _preprocess_candidate( cimg_library::CImg<float>& candidate, bool separator );

    const std::string _dump_plate();

private:

    std::vector<segment_status> m_segment_status;

    vectorF m_num_output;
    vectorF m_let_output;

    boost::shared_ptr<neurocl::sample> m_sample;
    neurocl::network_manager& m_net_num;
    neurocl::network_manager& m_net_let;
};

}; //namespace alpr

#endif //PLATE_RESOLUTION_H
