#ifndef NETWORK_INTERFACE_H
#define NETWORK_INTERFACE_H

#include <vector>

namespace neurocl {

class network_interface
{
public:

    // Convention : input layer is index 0
    virtual void add_layers_2d( const std::vector<size_t>& layer_sizes ) = 0;

    virtual void set_input_sample(  const size_t& isample_size, const float* isample,
                                    const size_t& osample_size, const float* osample ) = 0;

    virtual void feed_forward() = 0;
    virtual void gradient_descent() = 0;

    virtual const float output() = 0;
};

} //namespace neurocl

#endif //NETWORK_INTERFACE_H
