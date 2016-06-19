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

#include "lenet_bnu.h"
#include "lenet_manager.h"
#include "network_exception.h"
#include "network_file_handler.h"
#include "samples_manager.h"

#include <boost/chrono.hpp>
#include <boost/foreach.hpp>
#include <boost/make_shared.hpp>

//#define TRAIN_CHRONO

namespace neurocl {

lenet_manager::lenet_manager()
{
    m_net = boost::make_shared<lenet_bnu>();
    m_net->add_layers_2d( std::vector<layer_size>() ); // STUBBED FOR NOW
}

void lenet_manager::prepare_training_iteration()
{
    m_net->prepare_training();
}

void lenet_manager::finalize_training_iteration()
{
    m_net->gradient_descent();
}

void lenet_manager::train( const sample& s )
{
    _train( s );
}

void lenet_manager::train( const std::vector<sample>& training_set )
{
    size_t index = 0;

    BOOST_FOREACH( const neurocl::sample& s, training_set )
    {
        //std::cout << "lenet_manager::train - training sample " << (index+1) << "/" << training_set.size() << std::endl;

        _train( s );

        ++index;
    }
}

void lenet_manager::batch_train( 	const samples_manager& smp_manager,
									const size_t& epoch_size,
									const size_t& batch_size,
									t_progress_fct progress_fct )
{
    size_t progress_size = 0;
    const size_t pbm_size = epoch_size * smp_manager.samples_size();

    for ( size_t i=0; i<epoch_size; i++ )
    {
        std::cout << std::endl << "lenet_manager::batch_train - EPOCH " << (i+1) << "/" << epoch_size << std::endl;

        while ( true )
        {
            std::vector<neurocl::sample> samples = smp_manager.get_next_batch( batch_size );

            // end of training set management
            if ( samples.empty() )
                break;

            prepare_training_iteration();
            train( samples );
            finalize_training_iteration();

            progress_size += samples.size();

			int progress = ( ( 100 * progress_size ) / pbm_size );

			if ( progress_fct )
				progress_fct( progress );

            std::cout << "\rlenet_manager::batch_train - progress " << progress << "%";// << std::endl;
        }

        smp_manager.rewind();
    }

    std::cout << std::endl;

    save_network();
}

void lenet_manager::_train( const sample& s )
{
#ifdef TRAIN_CHRONO
    namespace bc = boost::chrono;
    bc::system_clock::time_point start = bc::system_clock::now();
    bc::milliseconds duration;
#endif

    // set input/output
    m_net->set_input( s.isample_size, s.isample );
    m_net->set_output( s.osample_size, s.osample );

    // forward/backward propagation
    m_net->feed_forward();
    m_net->back_propagate();

#ifdef TRAIN_CHRONO
    duration = boost::chrono::duration_cast<bc::milliseconds>( bc::system_clock::now() - start );
    std::cout << "lenet_manager::_train - training successfull in "  << duration.count() << "ms"<< std::endl;
#endif
}

void lenet_manager::compute_output( sample& s )
{
    m_net->set_input( s.isample_size, s.isample );
    m_net->feed_forward();
    output_ptr output_layer = m_net->output();
    std::copy( output_layer.outputs.get(), output_layer.outputs.get() + output_layer.num_outputs, const_cast<float*>( s.osample ) );
}

}; //namespace neurocl
