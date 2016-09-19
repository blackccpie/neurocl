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

#include "alpr.h"

#include <boost/lexical_cast.hpp>

#include <iostream>

#define NEUROCL_EPOCH_SIZE 100
#define NEUROCL_BATCH_SIZE 10
#define MAX_MATCH_ERROR 0.1f

using namespace neurocl;

int main( int argc, char *argv[] )
{
    std::cout << "Welcome to alpr!" << std::endl;

    if ( argc == 1 )
    {
        std::cout << "Invalid arguments!" << std::endl;
        std::cout << "example training numbers: ./alpr 1 alpr-train-num.txt topology-alpr-num.txt weights-alpr-num.bin" << std::endl;
        std::cout << "example training letters: ./alpr 1 alpr-train-let.txt topology-alpr-let.txt weights-alpr-let.bin" << std::endl;
        std::cout << "example testing: ./alpr 0 plaque.png topology-alpr-num.txt weights-alpr-num.bin topology-alpr-let.txt weights-alpr-let.bin" << std::endl;
        return -1;
    }

    try
    {
        // TODO : check command arguments with boost
        // 0 testing
        // 1 training
        bool training_enabled = ( boost::lexical_cast<int>( argv[1] ) == 1 );

        //************************* TRAINING *************************//

        if ( training_enabled )
        {
            /******** TRAIN ********/

            std::shared_ptr<network_manager_interface> net_manager = network_factory::build( network_factory::NEURAL_IMPL_MLP );
            net_manager->load_network( argv[3], argv[4] );

            samples_manager& smp_manager = samples_manager::instance();
            samples_manager::instance().load_samples( argv[2] );

            if ( argc == 6 )
                net_manager->batch_train( smp_manager, boost::lexical_cast<int>( argv[5] ), NEUROCL_BATCH_SIZE );
            else
            	net_manager->batch_train( smp_manager, NEUROCL_EPOCH_SIZE, NEUROCL_BATCH_SIZE );

            /******** VALIDATE ********/

            const std::vector<sample>& training_samples = smp_manager.get_samples();

            float mean_rmse = 0.f;
            size_t _rmse_score = 0;
            size_t _classif_score = 0;

            for ( size_t i = 0; i<training_samples.size(); i++ )
            {
                test_sample tsample( smp_manager.get_samples()[i] );
                net_manager->compute_output( tsample );

                std::cout << tsample.output() << std::endl;
                std::cout << tsample.ref_output() << std::endl;
                std::cout << tsample.RMSE() << std::endl;

                mean_rmse += tsample.RMSE();

                if ( tsample.RMSE() < MAX_MATCH_ERROR )
                    ++ _rmse_score;

                if ( tsample.classified() )
                    ++_classif_score;

            	//std::cout << "TEST OUTPUT IS : " << tsample.output() << std::endl;
            }

            mean_rmse /= static_cast<float>( training_samples.size() );

            std::cout << "MEAN RMSE IS " << mean_rmse << std::endl;
            std::cout << "RMSE SCORE IS " << _rmse_score << "/" << training_samples.size() << std::endl;
            std::cout << "CLASSIF SCORE IS " << _classif_score << "/" << training_samples.size() << std::endl;
        }

        //************************* TESTING *************************//

        else
        {
            std::shared_ptr<network_manager_interface> net_num = network_factory::build( network_factory::NEURAL_IMPL_MLP );
            net_num->load_network( argv[3], argv[4] );

            std::shared_ptr<network_manager_interface> net_let = network_factory::build( network_factory::NEURAL_IMPL_MLP );
            net_let->load_network( argv[5], argv[6] );

            alpr::license_plate lic( argv[2], net_num, net_let );
            lic.analyze();
        }
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

    std::cout << "Bye bye test_alpr!" << std::endl;

    return 0;
}
