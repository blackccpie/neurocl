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

#include "network.h"
#include "tensor_solver.h"
#include "tensor_utils.h"
#include "tensor_loss_functions.h"
#include "tensor_gradient_checker.h"
#include "conv_layer.h"
#include "full_layer.h"
#include "pool_layer.h"
#include "input_layer.h"
#include "output_layer.h"
#include "dropout_layer.h"

#include <boost/range/adaptor/reversed.hpp>

namespace neurocl { namespace convnet {

std::atomic_size_t network::m_training_samples{ 0 };

//#define VERBOSE_NETWORK

network::network()
{
    m_solver = tensor_solver_factory::build();
}

network::~network()
{
    //dump_image_features();
}

void network::set_training( bool training )
{
    layer::set_training( training );
}

void network::add_layers( const std::vector<layer_descr>& layers )
{
    size_t conv_idx = 0;
    size_t pool_idx = 0;
    size_t drop_idx = 0;
    size_t full_idx = 0;

    for ( auto& _layer : layers )
    {
        std::shared_ptr<layer> l;
        switch( _layer.type )
        {
        case INPUT_LAYER:
            {
                std::shared_ptr<input_layer> in = std::make_shared<input_layer>();
                in->populate( _layer.sizeX, _layer.sizeY, _layer.sizeZ );
                l = in;
            }
            break;
        case CONV_LAYER:
            {
                std::shared_ptr<conv_layer_iface> c =
                    std::make_shared< conv_layer<tensor_activations::sigmoid> >( "c" + std::to_string(++conv_idx) );
                c->set_filter_size( _layer.sizeF );
                c->populate( m_layers.back(), _layer.sizeX, _layer.sizeY, _layer.sizeZ );
                l = c;
            }
            break;
        case POOL_LAYER:
            {
                std::shared_ptr<pool_layer> s = std::make_shared<pool_layer>( "s" + std::to_string(++pool_idx) );
                s->populate( m_layers.back(), _layer.sizeX, _layer.sizeY, _layer.sizeZ );
                l = s;
            }
            break;
        case FULL_LAYER:
            {
                std::shared_ptr<full_layer_iface> f =
                    std::make_shared< full_layer<tensor_activations::sigmoid> >( "f" + std::to_string(++full_idx) );
                f->populate( m_layers.back(), _layer.sizeX, _layer.sizeY, _layer.sizeZ );
                l = f;
            }
            break;
        case OUTPUT_LAYER:
            {
                std::shared_ptr<output_layer_iface> out =
                    std::make_shared< output_layer<tensor_activations::sigmoid,tensor_loss_functions::cross_entropy> >();
                out->populate( m_layers.back(), _layer.sizeX, _layer.sizeY, _layer.sizeZ );
                l = out;
            }
            break;
        case DROPOUT_LAYER:
            {
                std::shared_ptr<dropout_layer> d = std::make_shared<dropout_layer>( "d" + std::to_string(++drop_idx) );
                d->populate( m_layers.back(), _layer.sizeX, _layer.sizeY, _layer.sizeZ );
                l = d;
            }
            break;
        }
        m_layers.emplace_back( l );
    }
}

void network::set_input(  const size_t& in_size, const float* in )
{
    // TODO-CNN : for now works only because input layer has no depth for now!

    std::shared_ptr<layer> layer;
    std::shared_ptr<input_layer> input_layer =
        std::dynamic_pointer_cast<neurocl::convnet::input_layer>( m_layers.front() );

    if ( in_size > input_layer->size() )
        throw network_exception( "sample size exceeds allocated layer size!" );

#ifdef VERBOSE_NETWORK
    LOGGER(info) << "network::set_input - input (" << in << ") size = " << in_size << std::endl;
#endif

    input_layer->fill( 0, 0, in_size, in );
}

void network::set_output( const size_t& out_size, const float* out )
{
    // TODO-CNN : for now works only because output layer has no depth for now!

    std::shared_ptr<output_layer_iface> output_layer =
        std::dynamic_pointer_cast<neurocl::convnet::output_layer_iface>( m_layers.back() );

    if ( out_size > output_layer->size() )
        throw network_exception( "output size exceeds allocated layer size!" );

#ifdef VERBOSE_NETWORK
    LOGGER(info) << "network::set_output - output (" << out << ") size = " << out_size << std::endl;
#endif

    output_layer->fill( 0, 0, out_size, out );
}

const layer_ptr network::get_layer_ptr( const size_t layer_idx )
{
    if ( layer_idx >= m_layers.size() )
	{
        LOGGER(error) << "network::get_layer_ptr - cannot access layer " << layer_idx << std::endl;
        throw network_exception( "invalid layer index" );
    }

    std::shared_ptr<layer> _layer = m_layers[layer_idx];

    LOGGER(info) << "network::get_layer_ptr - getting layer  " << _layer->type() << std::endl;

    layer_ptr l( _layer->nb_weights(), _layer->nb_bias() );
    _layer->fill_w( l.weights.get() );
    _layer->fill_b( l.bias.get() );

    return l;
}

void network::set_layer_ptr( const size_t layer_idx, const layer_ptr& l )
{
    if ( layer_idx >= m_layers.size() )
    {
        LOGGER(error) << "network::set_layer_ptr - cannot access layer " << layer_idx << std::endl;
        throw network_exception( "invalid layer index" );
    }

    std::shared_ptr<layer> _layer = m_layers[layer_idx];

    if ( ( _layer->nb_weights() != l.num_weights ) || ( _layer->nb_bias() != l.num_bias ) )
    {
        LOGGER(error) << "network::set_layer_ptr - inconsistent layer " << layer_idx << " size" << std::endl;
        throw network_exception( "inconsistent layer size" );
    }

    LOGGER(info) << "network::set_layer_ptr - setting layer  " << _layer->type() << std::endl;

    _layer->fill_w( _layer->nb_weights(), l.weights.get() );
    _layer->fill_b( _layer->nb_bias(), l.bias.get() );
}

const output_ptr network::output()
{
    // TODO-CNN : for now works only because last layer has no depth for now

    std::shared_ptr<output_layer_iface> output_layer =
        std::dynamic_pointer_cast<neurocl::convnet::output_layer_iface>( m_layers.back() );

    output_ptr o( output_layer->width() * output_layer->height() );
    output_layer->fill( 0, 0, o.outputs.get() );

    return std::move( o );
}

void network::clear_gradients()
{
    for ( auto _layer : m_layers )
    {
        _layer->clear_gradients();
    }

    m_training_samples = 0;
}

void network::feed_forward()
{
    for ( auto _layer : m_layers )
    {
#ifdef VERBOSE_NETWORK
        std::cout << "--> feed forwarding " << _layer->type() << " layer" << std::endl;
#endif
        _layer->feed_forward();
    }
}

void network::back_propagate()
{
    for ( auto _layer : boost::adaptors::reverse( m_layers ) )
    {
#ifdef VERBOSE_NETWORK
        std::cout << "--> back propagating " << _layer->type() << " layer" << std::endl;
#endif
        _layer->back_propagate();
    }

    for ( auto _layer : m_layers )
    {
#ifdef VERBOSE_NETWORK
        std::cout << "--> updating gradients " << _layer->type() << " layer" << std::endl;
#endif
        _layer->update_gradients();
    }

    ++m_training_samples;
}

void network::gradient_descent()
{
    m_solver->set_size( m_training_samples );

    for ( auto _layer : m_layers )
    {
        _layer->gradient_descent( m_solver );
    }
}

void network::gradient_check( const output_ptr& out_ref )
{
    // NOTE : network input & output have to be previously set!!

    auto err_func = []( output_ptr a, output_ptr b ) {
        float _acc = 0.f;
        float* pa = a.outputs.get();
        float* pb = b.outputs.get();
        for ( size_t i=0; i<a.num_outputs; i++ )
            _acc += (pa[i] - pb[i])*(pa[i] - pb[i]);

        return _acc / ( 2.f * static_cast<float>( a.num_outputs ) ); // TODO : mse error, use a generic loss class?
     };

    float epsilon = 1e-4f;

    for ( auto _layer : m_layers )
    {
        std::unique_ptr<tensor_gradient_checker> grad_check = _layer->get_gradient_checker();
        if ( !grad_check )
            continue;

        for ( size_t i = 0; i<grad_check->size(); i++ )
        {
            // Store weight
            grad_check->store();

            // Compute output with +epsilon increment
            grad_check->mod( +epsilon );
            feed_forward();
            back_propagate();

            const output_ptr out_p = output();
            float err_p = err_func( out_p, out_ref );

            // Compute output with -epsilon increment
            grad_check->mod( -epsilon );
            feed_forward();
            back_propagate();

            const output_ptr out_m = output();
            float err_m = err_func( out_m, out_ref );

            grad_check->set_grad( ( err_p - err_m ) / ( 2.f * epsilon ) );

            // Restore weight
            grad_check->restore();

            // switch to next parameter
            grad_check->next();

            clear_gradients();
        }

        // compute "standard" deltas
        feed_forward();
        back_propagate();

        LOGGER(info) << "network::gradient_check - layer " << _layer->type() << " gradient check error : " << grad_check->error() << std::endl;
    }
}

void network::dump_image_features()
{
    //tensor_utils::visualizer::dump_features( "features_dump", m_layers[1]->type(), m_layers[1]->weights({}) );
    //return;

    for ( auto _layer : m_layers )
    {
        // dump only if the layer has weights
        if ( _layer->nb_weights() )
        	tensor_utils::visualizer::dump_features( "features_dump", _layer->type(), _layer->weights({}) );
    }
}

} /*namespace neurocl*/ } /*namespace convnet*/
