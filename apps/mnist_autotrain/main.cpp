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

#include <boost/program_options.hpp>

#include <iostream>
#include <fstream>

#define NEUROCL_MAX_EPOCH_SIZE 1000
#define NEUROCL_BATCH_SIZE 10
#define NEUROCL_STOPPING_SCORE 99.f

using namespace neurocl;
namespace po = boost::program_options;

console_color::modifier c_red(console_color::color_code::FG_RED);
console_color::modifier c_green(console_color::color_code::FG_GREEN);
console_color::modifier c_def(console_color::color_code::FG_DEFAULT);

float compute_score(	const int i,
                    	const samples_manager& smp_manager,
                    	const std::shared_ptr<network_manager_interface>& net_manager,
                        float& rmse,
                    	float& last_rmse,
                        bool testing )
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

    float score = static_cast<float>( 1000 * _classif_score / training_samples.size() ) / 10.f;
    rmse = mean_rmse / static_cast<float>( training_samples.size() );

    console_color::modifier* mod = ( rmse <= last_rmse ) ? &c_green : &c_red;

    std::cout << "EPOCH " << i << " - CURRENT " << ( testing ? "TESTING" : "TRAINING" ) << " SCORE IS : " << score << "% (" << _classif_score << "/" << training_samples.size() << ") "
        << "CURRENT RMSE IS : " << rmse << " (" << *mod << (rmse-last_rmse) << c_def << ")" << std::endl;

    last_rmse = rmse;

    return score;
}

int main( int argc, char *argv[] )
{
    std::cout << "Welcome to mnist_autotrain!" << std::endl;

    size_t train_restrict = 0;
    size_t valid_restrict = 0;
    bool scheduling = false;
    bool gradient_check = false;

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message")
        ("scheduling,s", po::value<bool>( &scheduling )->default_value( false ), "enable scheduling")
        ("gradient_check,g", po::value<bool>( &gradient_check )->default_value( false ), "enable gradient check mode")
        ("train_restrict,t", po::value<size_t>( &train_restrict )->default_value( 0 ), "restricted training dataset size")
        ("valid_restrict,v", po::value<size_t>( &valid_restrict )->default_value( 0 ), "restricted validation dataset size")
    ;

    po::variables_map vm;
    po::store( po::parse_command_line( argc, argv, desc ), vm );
    po::notify( vm );

    if ( vm.count( "help" ) )
    {
        std::cout << desc << std::endl;
        return 0;
    }

    std::cout << std::endl;
    std::cout << "===== mnist autotrain options recap =====" << std::endl;
    if ( vm.count( "scheduling" ) )
        std::cout << "scheduling has been " << ( scheduling ? "enabled" : "disabled" ) << std::endl;
    if ( vm.count( "gradient_check" ) )
        std::cout << "gradient_check mode has been " << ( gradient_check ? "enabled" : "disabled" ) << std::endl;
    if ( vm.count( "train_restrict" ) )
        std::cout << "training dataset size has been restricted to " << train_restrict << std::endl;
    if ( vm.count( "valid_restrict" ) )
        std::cout << "validation dataset size has been restricted to " << valid_restrict << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << std::endl;

    logger_manager& lm = logger_manager::instance();
    lm.add_logger( policy_type::cout, "mnist_autotrain" );

    try
    {
        samples_manager smp_train_manager;
        if ( train_restrict )
            smp_train_manager.restrict_dataset( train_restrict );
        smp_train_manager.load_samples( "../nets/mnist/training/mnist-train.txt" );

        samples_manager smp_validate_manager;
        if ( valid_restrict )
            smp_validate_manager.restrict_dataset( valid_restrict );
        smp_validate_manager.load_samples( "../nets/mnist/training/mnist-validate.txt" );

        std::shared_ptr<network_manager_interface> net_manager = network_factory::build();
        net_manager->load_network( "../nets/mnist/topology-mnist-lenet-drop.txt", "../nets/mnist/weights-mnist-lenet.bin" );

        if ( gradient_check )
        {
            net_manager->gradient_check( smp_train_manager.get_samples()[0] );
            return 0;
        }

        neurocl::learning_scheduler& sched = neurocl::learning_scheduler::instance();
        sched.enable_scheduling( scheduling );

        float train_score = 0.f;
        float valid_score = 0.f;
        float last_train_rmse = 0.f;
        float last_valid_rmse = 0.f;
        float rmse = 0.f;

        std::ofstream output_file( "mnist_training.csv", std::fstream::app );

        for ( int i=0; i<NEUROCL_MAX_EPOCH_SIZE; i++ )
        {
            net_manager->batch_train( smp_train_manager, 1, NEUROCL_BATCH_SIZE );

            train_score = compute_score( i, smp_train_manager, net_manager, rmse, last_train_rmse, false );
            valid_score = compute_score( i, smp_validate_manager, net_manager, rmse, last_valid_rmse, true );

            sched.push_error( rmse );

            output_file << (i+1) << ','
                        << train_score << ','
                        << valid_score << ','
                        << rmse << ','
                        << sched.get_learning_rate() << '\n';
            output_file.flush();

            if ( valid_score > NEUROCL_STOPPING_SCORE )
            {
                std::cout << "TRAINING SUCCEEDED IN " << (i+1) << " EPOCHS :-)" << std::endl;
                return 1;
            }
        }

        std::cout << "VALIDATION SCORE IS : " << valid_score << "% AFTER " << NEUROCL_MAX_EPOCH_SIZE << " EPOCHS :-(" << std::endl;
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
