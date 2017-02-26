/*
The MIT License

Copyright (c) 2015-2017 Albert Murienne

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

#ifndef NETWORK_RANDOM_H
#define NETWORK_RANDOM_H

#include <common/network_config.h>

#include <random>

namespace neurocl {

namespace random {

class seed
{
public:
    static seed& instance()
    {
        static seed s;
        return s;
    }
    unsigned int operator()()
    {
        static unsigned int _offset = 0;

        if ( m_seed )
            return m_seed.get() + _offset++;
        else
			return m_rd();
    }
private:
    seed()
    {
        m_seed = network_config::instance().get_param<unsigned int>( "random_seed" );
    }
    virtual ~seed() {}
private:

	// optional seed
    boost::optional<unsigned int> m_seed;

    // using random_device allows to have different random sets at each runtime
    std::random_device m_rd;
};

class rand_bernoulli_generator
{
public:
    rand_bernoulli_generator( const float p )
        : m_rng{ seed::instance()() }, m_bernoulli{ p } {}
    virtual ~rand_bernoulli_generator() {}

    template <typename T = bool>
    T gen() { return static_cast<T>( m_bernoulli( m_rng ) ); }

private:
    std::mt19937 m_rng;
    std::bernoulli_distribution m_bernoulli;
};

class rand_gaussian_generator
{
public:
    rand_gaussian_generator( const float mean, const float stddev )
        : m_rng{ seed::instance()() }, m_normal{ mean, stddev }
    {
        //m_rng.seed(...);
    }

    float operator()() { return m_normal( m_rng ); }

private:

    std::mt19937 m_rng;
    std::normal_distribution<> m_normal;
};

} //namespace random

} //namespace neurocl

#endif //NETWORK_UTILS_H
