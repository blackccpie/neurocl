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

#ifndef SAMPLES_MANAGER_H
#define SAMPLES_MANAGER_H

#include "network_sample.h"

#include <boost/function.hpp>
#include <boost/shared_array.hpp>

#include <vector>

namespace neurocl {

using t_preproc = boost::function<void (float*,const size_t,const size_t)>;

class samples_manager
{
public:

    static samples_manager& instance()
    {
        static samples_manager m;
        return m;
    }

    void load_samples( const std::string &input_filename, bool shuffle = false, t_preproc extra_preproc = t_preproc() );

    const size_t samples_size() const
    {
        return m_samples_set.size();
    }

    const std::vector<neurocl::sample>& get_samples() const
    {
        return m_samples_set;
    }

    const std::vector<neurocl::sample> get_next_batch( const size_t size ) const;

    void rewind() const;

private:

    samples_manager() : m_batch_index( 0 ), m_end( false ) {}
    virtual ~samples_manager() {}

private:

    mutable bool m_end;
    mutable size_t m_batch_index;

    std::vector< boost::shared_array<float> > m_input_samples;
    std::vector< boost::shared_array<float> > m_output_samples;
    std::vector<neurocl::sample> m_samples_set;
};

} //namespace neurocl

#endif //SAMPLES_MANAGER_H
