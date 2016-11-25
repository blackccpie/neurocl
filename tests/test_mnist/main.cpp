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

#include "neurocl.h"

#include <boost/lexical_cast.hpp>

#include <chrono>
#include <iostream>

#define NEUROCL_EPOCH_SIZE 30
#define NEUROCL_BATCH_SIZE 20
#define MAX_MATCH_ERROR 0.1f

using namespace neurocl;

int main( int argc, char *argv[] )
{
    std::cout << "Welcome to test_mnist!" << std::endl;

    logger_manager& lm = logger_manager::instance();
    lm.add_logger( policy_type::cout, "test_mnist" );

    if ( argc == 1 )
    {
        std::cout << "Invalid arguments!" << std::endl;
        std::cout << "example: ./test_mnist mnist-train.txt topology-mnist.txt weights-mnist.bin 10" << std::endl;
        return -1;
    }

    try
    {
        samples_manager& smp_manager = neurocl::samples_manager::instance();
        samples_manager::instance().load_samples( argv[1] );

        std::shared_ptr<network_manager_interface> net_manager = network_factory::build();
        net_manager->load_network( argv[2], argv[3] );

        //************************* TRAINING *************************//

        std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

        //net_manager.dump_weights();
        //net_manager.dump_bias();

        if ( argc == 5 )
            net_manager->batch_train( smp_manager, boost::lexical_cast<int>( argv[4] ), NEUROCL_BATCH_SIZE );
        else
            net_manager->batch_train( smp_manager, NEUROCL_EPOCH_SIZE, NEUROCL_BATCH_SIZE );

        std::chrono::milliseconds duration_training = std::chrono::duration_cast<std::chrono::milliseconds>( std::chrono::system_clock::now() - start );

        // Dump weights for debugging purposes
        //net_manager.dump_weights();
        //net_manager.dump_bias();
        //net_manager.dump_activations();

        //************************* TESTING *************************//

        start = std::chrono::system_clock::now();

        const std::vector<sample>& training_samples = smp_manager.get_samples();

        float mean_rmse = 0.f;
        size_t _rmse_score = 0;
        size_t _classif_score = 0;

        for ( size_t i = 0; i<training_samples.size(); i++ )
        {
            test_sample tsample( smp_manager.get_samples()[i] );
            net_manager->compute_output( tsample );

            //std::cout << tsample.output() << std::endl;
            //std::cout << tsample.ref_output() << std::endl;
            //std::cout << tsample.RMSE() << std::endl;

            mean_rmse += tsample.RMSE();

            if ( tsample.RMSE() < MAX_MATCH_ERROR )
                ++ _rmse_score;

            if ( tsample.classified() )
                ++_classif_score;

        	//std::cout << "TEST OUTPUT IS : " << tsample.output() << std::endl;
        }

        std::chrono::milliseconds duration_testing = std::chrono::duration_cast<std::chrono::milliseconds>( std::chrono::system_clock::now() - start );

        mean_rmse /= static_cast<float>( training_samples.size() );

        std::cout << "MEAN RMSE IS " << mean_rmse << std::endl;
        std::cout << "RMSE SCORE IS " << _rmse_score << "/" << training_samples.size()
            << " (" << static_cast<int>( 100 * _rmse_score / training_samples.size() ) << "%)" << std::endl;
        std::cout << "CLASSIF SCORE IS " << _classif_score << "/" << training_samples.size()
            << " (" << static_cast<int>( 100 * _classif_score / training_samples.size() ) << "%)" << std::endl;

        std::cout << "TRAINING DONE IN : " << duration_training.count() << "ms" << std::endl;
        std::cout << "TESTING DONE IN : " << duration_testing.count() << "ms" << std::endl;
    }
    catch( network_exception& e )
    {
        std::cerr << "network exception : " << e.what() << std::endl;
    }
    catch( std::exception& e )
    {
        std::cerr << "std::exception : " << e.what() << std::endl;
    }
    catch(...)
    {
        std::cerr << "unknown exception" << std::endl;
    }

    std::cout << "Bye bye test_mnist!" << std::endl;

    return 0;
}
