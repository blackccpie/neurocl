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

#ifndef NETWORK_RANDOM_H
#define NETWORK_RANDOM_H

//TODO-CNN : get rid of boost random
#include <boost/random.hpp>
#include <boost/random/random_device.hpp>
#include <boost/random/normal_distribution.hpp>

#include <random>

namespace neurocl {

namespace random {

class rand_bernoulli_generator
{
public:
    rand_bernoulli_generator( const float p )
        : m_gen( std::random_device{}() ), m_bernoulli{ p } {}
    virtual ~rand_bernoulli_generator() {}

    float operator()() { return m_bernoulli( m_gen ); }

private:
    std::mt19937 m_gen;
    std::bernoulli_distribution m_bernoulli;
};

class rand_gaussian_generator
{
public:
    rand_gaussian_generator( const float mean, const float stddev )
        : m_rng( m_rd ), m_nd( mean, stddev ), m_var_nor( m_rng, m_nd )
    {
        //m_var_nor.engine().seed( _seed() );
    }

    float operator()() { return m_var_nor(); }

private:

    // changing seed allows to have different random sets at each class instanciation,
    // but at each runtime the random sets will be the same
    //static int _seed() { static int i = 0; return i++; };

private:

    // using random_device allows to have different random sets at each runtime
    boost::random_device m_rd;
    boost::mt19937 m_rng;
    boost::normal_distribution<> m_nd;
    boost::variate_generator< boost::mt19937&,boost::normal_distribution<> > m_var_nor;
};

} //namespace random

} //namespace neurocl

#endif //NETWORK_UTILS_H
