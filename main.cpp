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

#include "network_manager.h"
#include "samples_manager.h"

#include <boost/foreach.hpp>

int main( int argc, char *argv[] )
{
    std::cout << "Welcome to neurocl!" << std::endl;

    try
    {
        // TODO : check command arguments with boost

        neurocl::samples_manager& smp_manager = neurocl::samples_manager::instance();
        neurocl::samples_manager::instance().load_samples( argv[1] );

        neurocl::network_manager net_manager( neurocl::network_manager::NEURAL_IMPL_BNU );
        net_manager.load_network( argv[2], "weights" );

        //************************* TRAINING *************************//

        for ( int i=0; i<30; i++ )
        {
            std::cout << "--> ITERATION " << (i+1) << "/30" << std::endl;

            std::vector<neurocl::sample> samples = smp_manager.get_next_batch( 10 );

            std::cout << "processing " << samples.size() << " samples" << std::endl;

            net_manager.prepare_training_iteration();

            BOOST_FOREACH( const neurocl::sample& sample, samples )
            {
                net_manager.train( sample );
            }

            net_manager.finalize_training_iteration();
        }

        // Dump weights for debugging purposes
        //net_manager.dump_weights();
        //net_manager.dump_activations();

        //************************* TESTING *************************//

        neurocl::sample& test_sample = smp_manager.get_samples()[679];
        net_manager.compute_output( test_sample );

        std::cout << "TEST OUTPUT IS : " << test_sample.output() << std::endl;
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

    std::cout << "Bye bye neurocl!" << std::endl;

    return 0;
}
