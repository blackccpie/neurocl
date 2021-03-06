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

#ifndef OUTPUT_LAYER_H
#define OUTPUT_LAYER_H

#include "layer.h"

#include "common/logger.h"

namespace neurocl { namespace convnet {

using nto = neurocl::convnet::tensor_operation;

namespace nta = neurocl::convnet::tensor_activations;

class output_layer_iface  : public layer
{
public:

    // populate layer
    virtual void populate(  const std::shared_ptr<layer>& prev_layer,
                            const size_t width,
                            const size_t height,
                            const size_t depth,
                            const size_t cache_size ) = 0;

    // get current loss
    virtual float loss() = 0;

    // fill with incoming buffer
    virtual void fill(  const size_t depth1,
                        const size_t depth2,
                        const size_t data_size,
                        const float* data ) = 0;

    // fill outcoming buffer
    virtual void fill(  const size_t depth1,
                        const size_t depth2,
                        float* data ) = 0;
};

template<class activationT,class errorT>
class output_layer final : public output_layer_iface
{
public:

    output_layer()
     :  m_weights( nullptr ), m_deltas_weights( nullptr ),
        m_bias( nullptr ), m_deltas_bias( nullptr ),
        m_prev_group_features( false )
    {
        static_assert( !std::is_same<activationT,tensor_activations::softmax_cross_entropy>::value ||
            ( std::is_same<activationT, tensor_activations::softmax_cross_entropy>::value &&
            std::is_same<errorT,tensor_loss_functions::cross_entropy_softmax>::value ),
            "softmax cross entropy activation must only be used with softmax cross entropy loss!" );
    }

    virtual ~output_layer() {}

    const std::string type() const override { return "output"; }

    tensor d_activation( const tensor& in ) const override { /* NOTHING TO DO */return tensor{}; }

    void populate(  const std::shared_ptr<layer>& prev_layer,
                    const size_t width,
                    const size_t height,
                    const size_t depth,
                    const size_t cache_size ) override
    {
        LOGGER(info) << "output_layer::populate - populating output layer" << std::endl;

        m_prev_layer = prev_layer;

        // check if we have to group previous layer feature maps
        // grouping is only allowed if current depth is 1
        if ( m_prev_layer->depth() != depth )
        {
            if ( depth > 1 )
                throw network_exception( "depth mismatch between output layer and previous layer" );
            else
                m_prev_group_features = true;
        }

        // TODO-CNN : no need to allocate in non-training mode!
        m_training_output.resize( width, height, 1, depth );

        m_feature_maps.resize( width, height, 1, depth );
        m_error_maps.resize( width, height, 1, depth );

		//http://neuralnetworksanddeeplearning.com/chap6.html
		// However, there's no particular reason the argument should apply to softmax layers.
		// So there's no a priori reason to apply that initialization again.
		// Rather than do that, I shall initialize all the weights and biases to be 0.
		// This is a rather ad hoc procedure, but works well enough in practice
        bool null_init = std::is_same<activationT,tensor_activations::softmax>();

        if ( null_init )
        {
            LOGGER(warning) << "output_layer::populate - ad hoc null init for softmax output activation" << std::endl;
        }

        if ( m_shared )
        {
            m_bias = tensor_tank::instance().get_shared( "bias", width, height, 1, depth );
            if ( !null_init ) m_bias->fill_random( 1 ); // stddev 1 for bias
            m_deltas_bias = tensor_tank::instance().get_cumulative( "bias_delta", width, height, 1, depth );
            m_bias_cache.resize( cache_size ); int i = 0;
            for ( auto& _bias : m_bias_cache )
                _bias = tensor_tank::instance().get_shared( "bias_cache" + std::to_string(i++), width, height, 1, depth );

            m_weights = tensor_tank::instance().get_shared( "weights", width * height, fan_in(), 1, depth );
            if ( !null_init ) m_weights->fill_random( fan_in() );
            m_deltas_weights = tensor_tank::instance().get_cumulative( "weights_delta", width * height, fan_in(), 1, depth );
            m_weights_cache.resize( cache_size ); int j = 0;
            for ( auto& _weights : m_weights_cache )
                _weights = tensor_tank::instance().get_shared( "weights_cache" + std::to_string(j++), width * height, fan_in(), 1, depth );
        }
        else
        {
        	m_bias = tensor_tank::instance().get_standard( "bias", width, height, 1, depth );
            if ( !null_init ) m_bias->fill_random( 1 ); // stddev 1 for bias
        	m_deltas_bias = tensor_tank::instance().get_standard( "bias_delta", width, height, 1, depth );
            m_bias_cache.resize( cache_size ); int i = 0;
            for ( auto& _bias : m_bias_cache )
                _bias = tensor_tank::instance().get_standard( "bias_cache" + std::to_string(i++), width, height, 1, depth );

        	m_weights = tensor_tank::instance().get_standard( "weights", width * height, fan_in(), 1, depth );
            if ( !null_init ) m_weights->fill_random( fan_in() );
        	m_deltas_weights = tensor_tank::instance().get_standard( "weights_delta", width * height, fan_in(), 1, depth );
            m_weights_cache.resize( cache_size ); int j = 0;
            for ( auto& _weights : m_weights_cache )
                _weights = tensor_tank::instance().get_standard( "weights_cache" + std::to_string(j++), width * height, fan_in(), 1, depth );
        }
    }

    size_t width() const override { return m_feature_maps.w(); }
    size_t height() const override { return m_feature_maps.h(); }
    size_t depth() const override { return m_feature_maps.d2(); }

    size_t nb_weights() const override { return m_weights->w() * m_weights->h() * m_weights->d1() * m_weights->d2(); }
    size_t nb_bias() const override { return m_bias->w() * m_bias->h() * m_bias->d1() * m_bias->d2(); }

    // fill with incoming buffer
    void fill(  const size_t depth1,
                const size_t depth2,
                const size_t data_size,
                const float* data ) override
    {
        m_training_output.fill( depth1, depth2, data_size, data );
    }

    // fill outcoming buffer
    void fill(  const size_t depth1,
                const size_t depth2,
                float* data ) override
    {
        m_feature_maps.fill( depth1, depth2, data );
    }

    const tensor& feature_maps() const override
        { return m_feature_maps; }

    void feed_forward() override
    {
        const auto& prev_feature_maps = m_prev_layer->feature_maps();

        if ( m_prev_group_features )
        {
            const tensor grouped_feature_maps = nto::group( prev_feature_maps );

            // apply weights and bias
            m_feature_maps = nto::muladd( *m_weights, grouped_feature_maps, *m_bias );
        }
        else
        {
        	// apply weights and bias
        	m_feature_maps = nto::muladd( *m_weights, prev_feature_maps, *m_bias );
        }

        // apply activation function
        activationT::f( m_feature_maps );
    }

    // one-hot activation output error
    template<typename U = activationT>
    typename std::enable_if<std::is_same<typename U::is_one_hot,std::true_type>::value,void>::type
    _compute_output_error()
    {
        m_error_maps = nto::elemul(
            activationT::d_f( m_feature_maps ),
            errorT::d_f( m_feature_maps, m_training_output )
        );
    }

    // non one-hot activation output error
    template<typename U = activationT>
    typename std::enable_if<!std::is_same<typename U::is_one_hot,std::true_type>::value,void>::type
    _compute_output_error()
    {
         m_error_maps = activationT::d_f(
             m_feature_maps,
             errorT::d_f( m_feature_maps, m_training_output )
         );
    }

    void back_propagate() override
    {
        const tensor& prev_feature_maps = m_prev_layer->feature_maps();
        tensor& prev_error_maps = m_prev_layer->error_maps({});

        // Need to back prop?
        if ( prev_error_maps.empty() )
            return;

		// compute current loss
        m_loss.add( m_feature_maps, m_training_output );

        // Compute errors

        // compute output layer error
        _compute_output_error();

        // compute previous layer error

        if ( m_prev_group_features )
        {
            const auto&& grouped_feature_maps = nto::group( prev_feature_maps );

            const tensor grouped_error_maps = nto::elemul(
                m_prev_layer->d_activation( grouped_feature_maps ),
                nto::multrans1( *m_weights, m_error_maps )
            );

            nto::ungroup( grouped_error_maps, prev_error_maps );
        }
        else
        {
    		prev_error_maps = nto::elemul(
        		m_prev_layer->d_activation( prev_feature_maps ),
        		nto::multrans1( *m_weights, m_error_maps )
    		);
        }
    }

    void update_gradients() override
    {
        // Compute gradients

        if ( m_prev_group_features )
        {
            const auto&& grouped_feature_maps = nto::group( m_prev_layer->feature_maps() );

            *m_deltas_weights += nto::multrans2( m_error_maps, grouped_feature_maps );
        	*m_deltas_bias += m_error_maps;
        }
        else
        {
        	*m_deltas_weights += nto::multrans2( m_error_maps, m_prev_layer->feature_maps() );
        	*m_deltas_bias += m_error_maps;
        }
    }

	void clear_gradients() override
    {
        m_deltas_weights->clear();
        m_deltas_bias->clear();

        m_loss.clear();
    }

    void gradient_descent( const std::shared_ptr<tensor_solver_iface>& solver ) override
    {
        // Optimize gradients

        nto::optimize<nto::optimize_mode::std>( solver, m_weights, m_weights_cache.data(), m_deltas_weights );
        nto::optimize<nto::optimize_mode::redux>( solver, m_bias, m_bias_cache.data(), m_deltas_bias );
    }

    float loss() override
    {
        return m_loss.mean();
    }

    // Fill weights
    void fill_w( const size_t data_size, const float* data ) override
    {
         m_weights->grouped_fill( data_size, data );
    }
    void fill_w( float* data ) override
    {
         m_weights->grouped_fill( data );
    }

    // Fill bias
    void fill_b( const size_t data_size, const float* data ) override
    {
         m_bias->grouped_fill( data_size, data );
    }
    void fill_b( float* data ) override
    {
         m_bias->grouped_fill( data );
    }

    //! get gradient checker
    std::unique_ptr<tensor_gradient_checker> get_gradient_checker() override
    {
        return std::unique_ptr<tensor_gradient_checker>(
            new tensor_gradient_checker( *m_weights, *m_deltas_weights ) );
    }

    tensor& error_maps( key_errors ) override
        { return m_error_maps; }

protected:

    size_t fan_in() const override
    {
        size_t k_group = m_prev_group_features ? m_prev_layer->depth() : 1;
        return k_group * m_prev_layer->width() * m_prev_layer->height();
    }

private:

    template<class _errorT>
    class mean_loss
    {
    public:
        mean_loss() : m_size{0}, m_total_loss{0.f} {}

        void add( const tensor& a, const tensor& b )
        {
            m_total_loss += _errorT::f( a, b );
    		m_size++;
        }

        void clear()
        {
            m_total_loss = 0.f;
            m_size = 0;
        }

        float mean()
        {
            return m_total_loss / static_cast<float>( m_size );
        }

    private:

        size_t m_size;
        float m_total_loss;
    };

private:

    mean_loss<errorT> m_loss;

    std::shared_ptr<layer> m_prev_layer;

    tensor m_training_output;
    tensor m_feature_maps;
    tensor m_error_maps;

    tensor* m_weights;
    tensor* m_deltas_weights;
    std::vector<tensor*> m_weights_cache;

    tensor* m_bias;
    tensor* m_deltas_bias;
    std::vector<tensor*> m_bias_cache;

    bool m_prev_group_features;
};

} /*namespace neurocl*/ } /*namespace convnet*/

#endif //OUTPUT_LAYER_H
