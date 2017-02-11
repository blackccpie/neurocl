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

#include "neurocl.h"

#include "alpr.h"

#include <boost/lexical_cast.hpp>

#include <iostream>
#include <map>

#define NEUROCL_EPOCH_SIZE 100
#define NEUROCL_BATCH_SIZE 10
#define MAX_MATCH_ERROR 0.1f

using namespace neurocl;

std::map<std::string,std::tuple<std::string,std::string,std::string>> g_map_nets
= { { "N", { "../nets/alpr/topology-alpr-num2.txt", "../nets/alpr/weights-alpr-num2.bin", "../nets/alpr/training/alpr-train-num.txt" } },
    { "L", { "../nets/alpr/topology-alpr-let2.txt", "../nets/alpr/weights-alpr-let2.bin", "../nets/alpr/training/alpr-train-let.txt" } } };

int main( int argc, char *argv[] )
{
    std::cout << "Welcome to alpr!" << std::endl;

    if ( argc != 3 )
    {
        std::cout << "Invalid arguments!" << std::endl;
        std::cout << "example training numbers for 10 epochs: ./alpr N 10" << std::endl;
        std::cout << "example training letters for 10 epochs: ./alpr L 10" << std::endl;
        std::cout << "example testing: ./alpr T plaque.png" << std::endl;
        return -1;
    }

    try
    {
        logger_manager& lm = logger_manager::instance();
        lm.add_logger( policy_type::cout, "alpr" );

        std::string run_type = argv[1];

        bool testing_enabled = ( run_type == "T" );

        //************************* TRAINING *************************//

        if ( !testing_enabled )
        {
            /******** TRAIN ********/

            if ( g_map_nets.find( run_type ) == g_map_nets.end() )
            {
                std::cout << "Invalid run type argument: " << run_type << std::endl;
                return -1;
            }

            std::cout << "-> alpr training with topology : " << std::get<0>( g_map_nets[run_type] ) << std::endl;

            std::shared_ptr<network_manager_interface> net_manager = network_factory::build();
            net_manager->load_network( std::get<0>( g_map_nets[run_type] ), std::get<1>( g_map_nets[run_type] ) );

            samples_manager smp_manager;
            smp_manager.load_samples( std::get<2>( g_map_nets[run_type] ) );

            if ( argc == 3 )
                net_manager->batch_train( smp_manager, boost::lexical_cast<int>( argv[2] ), NEUROCL_BATCH_SIZE );
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

            mean_rmse /= static_cast<float>( training_samples.size() );

            std::cout << "MEAN RMSE IS " << mean_rmse << std::endl;
            std::cout << "RMSE SCORE IS " << _rmse_score << "/" << training_samples.size() << std::endl;
            std::cout << "CLASSIF SCORE IS " << _classif_score << "/" << training_samples.size() << std::endl;
        }

        //************************* TESTING *************************//

        else
        {
            std::shared_ptr<network_manager_interface> net_num = network_factory::build();
            net_num->load_network( "../nets/alpr/topology-alpr-num2.txt", "../nets/alpr/weights-alpr-num2.bin" );

            std::shared_ptr<network_manager_interface> net_let = network_factory::build();
            net_let->load_network( "../nets/alpr/topology-alpr-let2.txt", "../nets/alpr/weights-alpr-let2.bin" );

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
