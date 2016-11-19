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

#include <iostream>
#include <fstream>

#define NEUROCL_MAX_EPOCH_SIZE 1000
#define NEUROCL_BATCH_SIZE 20
#define NEUROCL_STOPPING_SCORE 98

using namespace neurocl;

int compute_score(  const samples_manager& smp_manager,
                    const std::shared_ptr<network_manager_interface>& net_manager,
                    float& rmse )
{
    const std::vector<sample>& training_samples = smp_manager.get_samples();

    float mean_rmse = 0.f;
    size_t _classif_score = 0;

    for ( size_t i = 0; i<training_samples.size(); i++ )
    {
        test_sample tsample( smp_manager.get_samples()[i] );
        net_manager->compute_output( tsample );

        mean_rmse += tsample.RMSE();

        if ( tsample.classified() )
            ++_classif_score;

        tsample.restore_ref();
    }

    std::cout << "DETAILED SCORE IS " << _classif_score << "/" << training_samples.size() << std::endl;

    rmse = mean_rmse / static_cast<float>( training_samples.size() );

    return static_cast<int>( 100 * _classif_score / training_samples.size() );
}

int main( int argc, char *argv[] )
{
    std::cout << "Welcome to mnist_autotrain!" << std::endl;

    logger_manager& lm = logger_manager::instance();
    lm.add_logger( policy_type::cout, "mnist_autotrain" );

    /*if ( argc == 1 )
    {
        std::cout << "Invalid arguments!" << std::endl;
        std::cout << "example: ./mnist_autotrain" << std::endl;
        return -1;
    }*/

    try
    {
        samples_manager& smp_manager = neurocl::samples_manager::instance();
        //samples_manager::instance().load_samples( "../nets/mnist/training/mnist-train.txt" );
        samples_manager::instance().load_samples( "../nets/mnist/training/mnist-validate.txt" );

        std::shared_ptr<network_manager_interface> net_manager = network_factory::build();
        net_manager->load_network( "../nets/mnist/topology-mnist-lenet.txt", "../nets/mnist/weights-mnist-lenet.bin" );

        int score = 0;
        float rmse = 0.f;
        float last_rmse = 0.f;

        std::ofstream output_file( "mnist_training.csv" );

        for ( int i=0; i<NEUROCL_MAX_EPOCH_SIZE; i++ )
        {
            net_manager->batch_train( smp_manager, 1, NEUROCL_BATCH_SIZE );

            score = compute_score( smp_manager, net_manager, rmse );

            output_file << (i+1) << ',' << rmse << '\n';

            float rmse_diff = rmse - last_rmse;

            std::cout << "CURRENT SCORE IS : " << score << "% CURRENT RMSE IS : " << rmse << " (" << rmse_diff << ")" << std::endl;

            last_rmse = rmse;

            if ( score > NEUROCL_STOPPING_SCORE )
            {
                std::cout << "TRAINING SUCCEEDED IN " << (i+1) << " EPOCHS :-)" << std::endl;
                return 1;
            }
        }

        std::cout << "TRAINING SCORE IS : " << score << "% AFTER " << NEUROCL_MAX_EPOCH_SIZE << " EPOCHS :-(" << std::endl;
    }
    catch( neurocl::network_exception& e )
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

    std::cout << "Bye bye mnist_autotrain!" << std::endl;

    return 0;
}
