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
#include "console_color.h"

#include <iostream>
#include <fstream>

#define NEUROCL_MAX_EPOCH_SIZE 1000
#define NEUROCL_BATCH_SIZE 20
#define NEUROCL_STOPPING_SCORE 99.f

using namespace neurocl;

console_color::modifier c_red(console_color::color_code::FG_RED);
console_color::modifier c_green(console_color::color_code::FG_GREEN);
console_color::modifier c_def(console_color::color_code::FG_DEFAULT);

float compute_score(  const int i,
                    const samples_manager& smp_manager,
                    const std::shared_ptr<network_manager_interface>& net_manager,
                    float& rmse )
{
    const std::vector<sample>& training_samples = smp_manager.get_samples();

    static float last_rmse = 0.f;
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

    float score = static_cast<float>( 1000 * _classif_score / training_samples.size() ) / 10.f;
    rmse = mean_rmse / static_cast<float>( training_samples.size() );

    console_color::modifier* mod = ( rmse <= last_rmse ) ? &c_green : &c_red;

    std::cout << "EPOCH " << i << " - CURRENT SCORE IS : " << score << "% (" << _classif_score << "/" << training_samples.size() << ") "
        << "CURRENT RMSE IS : " << rmse << " (" << *mod << (rmse-last_rmse) << c_def << ")" << std::endl;

    last_rmse = rmse;

    return score;
}

int main( int argc, char *argv[] )
{
    std::cout << "Welcome to mnist_autotrain!" << std::endl;

    logger_manager& lm = logger_manager::instance();
    lm.add_logger( policy_type::cout, "mnist_autotrain" );

    bool gradient_check = ( ( argc == 2 ) && ( std::string( argv[1] ) == "-gc" ) );

    try
    {
        samples_manager smp_train_manager;
        //smp_train_manager.restrict_dataset( 100 );
        smp_train_manager.load_samples( "../nets/mnist/training/mnist-train.txt" );

        samples_manager smp_validate_manager;
        //smp_validate_manager.restrict_dataset( 100 );
        smp_validate_manager.load_samples( "../nets/mnist/training/mnist-validate.txt" );

        std::shared_ptr<network_manager_interface> net_manager = network_factory::build();
        net_manager->load_network( "../nets/mnist/topology-mnist-lenet.txt", "../nets/mnist/weights-mnist-lenet.bin" );

        if ( gradient_check )
        {
            net_manager->gradient_check( smp_train_manager.get_samples()[0] );
            return 0;
        }

        neurocl::learning_scheduler& sched = neurocl::learning_scheduler::instance();
        sched.enable_scheduling( true );

        float score = 0.f;
        float rmse = 0.f;

        std::ofstream output_file( "mnist_training.csv", std::fstream::app );

        for ( int i=0; i<NEUROCL_MAX_EPOCH_SIZE; i++ )
        {
            net_manager->batch_train( smp_train_manager, 1, NEUROCL_BATCH_SIZE );

            score = compute_score( i, smp_validate_manager, net_manager, rmse );

            sched.push_error( rmse );

            output_file << (i+1) << ',' << sched.get_learning_rate() << ',' << rmse << '\n';

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
