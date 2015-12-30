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
