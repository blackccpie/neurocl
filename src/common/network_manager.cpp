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

#include "network_manager.h"

#include "interfaces/network_interface.h"
#include "interfaces/network_file_handler_interface.h"

#include "common/network_exception.h"
#include "common/samples_manager.h"
#include "common/logger.h"

#include <chrono>
#include <iostream>

//#define TRAIN_CHRONO

namespace neurocl {

class scoped_training
{
public:
    scoped_training( std::shared_ptr<network_interface> net ) : m_net( net )
    {
        m_net->set_training( true );
    }
    virtual ~scoped_training()
    {
        m_net->set_training( false );
    }
private:
    std::shared_ptr<network_interface> m_net;
};

void network_manager::_assert_loaded()
{
    if ( !m_network_loaded )
        throw network_exception( "no network loaded!" );
}

void network_manager::set_training( bool training, key_training )
{
    m_net->set_training( training );
}

void network_manager::load_network( const std::string& topology_path, const std::string& weights_path )
{
    m_net_file_handler->load_network_topology( topology_path );
    m_net_file_handler->load_network_weights( weights_path );

    m_network_loaded = true;

    LOGGER(info) << "network_manager::load_network - network loaded" << std::endl;
}

void network_manager::save_network()
{
    _assert_loaded();

    m_net_file_handler->save_network_weights();
}

void network_manager::batch_train(	const samples_manager& smp_manager,
                                    const size_t& epoch_size,
                                    const size_t& batch_size,
                                    t_progress_fct progress_fct )
{
    _assert_loaded();

    scoped_training _scoped_training( m_net );

    std::shared_ptr<samples_augmenter> smp_augmenter;// = smp_manager.get_augmenter();

    size_t progress_size = 0;
    const size_t pbm_size = epoch_size * smp_manager.samples_size();

    for ( size_t i=0; i<epoch_size; i++ )
    {
        LOGGER(info) << "network_manager::batch_train - EPOCH " << (i+1) << "/" << epoch_size << std::endl;

        while ( true )
        {
            std::vector<neurocl::sample> samples = smp_manager.get_next_batch( batch_size );

            // end of training set management
            if ( samples.empty() )
                break;

            prepare_training_epoch();
            _train_batch( samples, smp_augmenter );
            finalize_training_epoch();

            progress_size += samples.size();

			int progress = ( ( 100 * progress_size ) / pbm_size );

			if ( progress_fct )
				progress_fct( progress );

            std::cout << "\rnetwork_manager::batch_train - progress " << progress << "%";// << std::endl;
        }

        std::cout << "\r";

        smp_manager.rewind();
        smp_manager.shuffle();
    }

    std::cout << std::endl;

    save_network();
}

void network_manager::prepare_training_epoch()
{
    m_net->clear_gradients();
}

void network_manager::finalize_training_epoch()
{
    m_net->gradient_descent();
}

void network_manager::train( const sample& s, key_training )
{
    _assert_loaded();

    _train_single( s );
}

void network_manager::_train_batch( const std::vector<sample>& training_set, const std::shared_ptr<samples_augmenter>& smp_augmenter )
{
    _assert_loaded();

    size_t index = 0;

    for( const auto& s : training_set )
    {
        //LOGGER(info) << "network_manager::_train_batch - training sample " << (index+1) << "/" << training_set.size() << std::endl;

        if ( !smp_augmenter )
        {
			_train_single( s );
        }
        else
        {
            sample _s = smp_augmenter->translate( s, samples_augmenter::rand_shift(), samples_augmenter::rand_shift() );
            _train_single( _s );
        }

    ++index;
    }
}

void network_manager::_train_single( const sample& s )
{
#ifdef TRAIN_CHRONO
    namespace sc = std::chrono;
    sc::system_clock::time_point start = sc::system_clock::now();
    sc::milliseconds duration;
#endif

    // set input/output
    m_net->set_input( s.isample_size, s.isample );
    m_net->set_output( s.osample_size, s.osample );

    // forward/backward propagation
    m_net->feed_forward();
    m_net->back_propagate();

#ifdef TRAIN_CHRONO
    duration = sc::duration_cast<sc::milliseconds>( sc::system_clock::now() - start );
    LOGGER(info) << "network_manager::_train_single - training successfull in "  << duration.count() << "ms"<< std::endl;
#endif
}

void network_manager::compute_augmented_output( sample& s, const std::shared_ptr<samples_augmenter>& smp_augmenter )
{
	// ONLY ROTATION AUGMENTATION IS IMPLEMENTED YET

    std::vector<int> rotations{ -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5 };

    std::vector<output_ptr> outputs;

    for ( auto& rot : rotations )
    {
        sample _s = smp_augmenter->rotate( s, rot );
        m_net->set_input( _s.isample_size, _s.isample );
        m_net->feed_forward();
        outputs.emplace_back( m_net->output() );
        //output_layer += m_net->output();
    }

    // TODO-CNN : not very proud of the efficiency of this code section...
    // still it is temporary as computing a mean image alos makes sense!

    int i = 0;
    int l = 0;
    float max = std::numeric_limits<float>::min();
    for ( const auto& output : outputs )
    {
        float _tmp_max = output.max_comp_val();

        std::cout << "network_manager::compute_augmented_output - " << _tmp_max << " " << output.max_comp_idx() << std::endl;

        if ( _tmp_max > max )
        {
            l = i;
            max = _tmp_max;
        }
        i++;
    }

    std::copy( outputs[l].outputs.get(), outputs[l].outputs.get() + outputs[l].num_outputs, const_cast<float*>( s.osample ) );
}

void network_manager::compute_output( sample& s )
{
    _assert_loaded();

    m_net->set_input( s.isample_size, s.isample );
    m_net->feed_forward();
    output_ptr output_layer = m_net->output();
    std::copy( output_layer.outputs.get(), output_layer.outputs.get() + output_layer.num_outputs, const_cast<float*>( s.osample ) );
}

void network_manager::compute_output( std::vector<sample>& s )
{
    _assert_loaded();

    // NOT IMPLEMENTED YET
    //m_net->batch_feed_forward( std::vector<input_ptr>&, std::vector<output_ptr>& );
}

void network_manager::gradient_check( const sample& s )
{
    _assert_loaded();

    m_net->set_input( s.isample_size, s.isample );
    m_net->set_output( s.osample_size, s.osample );

    output_ptr out_ref( s.osample_size );
    std::copy( s.osample, s.osample + s.osample_size, out_ref.outputs.get() );

    m_net->gradient_check( out_ref );
}

void network_manager::dump_weights()
{
    _assert_loaded();

    std::cout << m_net->dump_weights();
}

void network_manager::dump_bias()
{
    _assert_loaded();

    std::cout << m_net->dump_bias();
}

void network_manager::dump_activations()
{
    _assert_loaded();

    std::cout << m_net->dump_activations();
}

} /*namespace neurocl*/
