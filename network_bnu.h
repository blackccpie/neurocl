#ifndef NETWORK_BNU_H
#define NETWORK_BNU_H

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>

#include <vector>

typedef typename boost::numeric::ublas::vector<float> vectorF;
typedef typename boost::numeric::ublas::matrix<float> matrixF;

namespace neurocl {

class layer_bnu
{
public:

    layer_bnu();
	virtual ~layer_bnu() {}

    void populate( const size_t& size, const size_t& next_layer_size );

    vectorF& bias() { return m_bias; }
    vectorF& activations() { return m_activations; }
    matrixF& weights() { return m_output_weights; }
    vectorF& errors() { return m_errors; }
    matrixF& w_deltas() { return m_deltas_weight; }
    vectorF& b_deltas() { return m_deltas_bias; }

private:

    vectorF m_activations;
    vectorF m_errors;
    vectorF m_bias;
    vectorF m_deltas_bias;

    // We follow stanford convention:
    // http://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf
    matrixF m_output_weights;
    matrixF m_deltas_weight;
};

class network_bnu
{
public:

	network_bnu();
	virtual ~network_bnu() {}

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

    vectorF m_training_output;

    std::vector<layer_bnu> m_layers;
};

} //namespace neurocl

#endif //NETWORK_BNU_H
