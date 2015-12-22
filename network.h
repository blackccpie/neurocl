#ifndef NETWORK_H
#define NETWORK_H

#include <vexcl/vexcl.hpp>

namespace neurocl {

class layer
{
public:

    layer();
	virtual ~layer() {}

    void populate( const size_t& size, const size_t& next_layer_size );

    vex::vector<float>& bias() { return m_bias; }
    vex::vector<float>& activations() { return m_activations; }
    vex::vector<float>& weights() { return m_output_weights; }
    vex::vector<float>& errors() { return m_errors; }
    vex::vector<float>& w_deltas() { return m_deltas_weight; }
    vex::vector<float>& b_deltas() { return m_deltas_bias; }

    std::pair<size_t,size_t>& w_size() { return m_weights_size; }

private:

    std::pair<size_t,size_t> m_weights_size;

    vex::vector<float> m_activations;
    vex::vector<float> m_errors;
    vex::vector<float> m_bias;
    vex::vector<float> m_deltas_bias;

    // We follow stanford convention:
    // http://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf
    vex::vector<float> m_output_weights;
    vex::vector<float> m_deltas_weight;
};

class network
{
public:

	network();
	virtual ~network() {}

    // Convention : input layer is index 0
    void add_layers_2d( const std::vector<size_t>& layer_sizes );

    void set_training_sample(   const size_t& isample_size, const float* isample,
                                const size_t& osample_size, const float* osample );

    void feed_forward();
    void gradient_descent();

    const float output();

private:

    void _back_propagate();
    void _gradient_descent();

private:

    float m_learning_rate;  // [0.0..1.0]
    float m_weight_decay;   // [0.0..1.0]

    vex::vector<float> m_training_output;

    std::vector<layer> m_layers;
};

} //namespace neurocl

#endif //NETWORK_H
