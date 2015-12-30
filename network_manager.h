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

#ifndef NETWORK_MANAGER_H
#define NETWORK_MANAGER_H

#include <string>
#include <vector>

#include <boost/shared_array.hpp>

namespace neurocl {

class network_interface;

struct sample
{
    sample( const size_t isize, const float* idata, const size_t osize, float* odata )
        : isample_size( isize ), isample( idata ), osample_size( osize ), osample( odata ) {}

    size_t isample_size;
    const float* isample;
    size_t osample_size;
    float* osample;
};

class network_manager
{
public:

    typedef enum
    {
        NEURAL_IMPL_BNU = 0,
        NEURAL_IMPL_VEXCL
    } t_neural_impl;

public:

    network_manager( const t_neural_impl& impl );
	virtual ~network_manager() {}

    void load_network( const std::string& name );
    void save_network();

    void train( const std::vector<sample>& training_set );

    void compute_output( const sample& s );

private:

    bool m_network_loaded;

    boost::shared_ptr<network_interface> m_net;
};

} //namespace neurocl

#endif //NETWORK_MANAGER_H
