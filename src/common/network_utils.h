#ifndef NETWORK_UTILS_H
#define NETWORK_UTILS_H

#include <boost/random.hpp>
#include <boost/random/random_device.hpp>
#include <boost/random/normal_distribution.hpp>

namespace neurocl {

namespace utils {

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

} //namespace utils

} //namespace neurocl

#endif //NETWORK_UTILS_H
