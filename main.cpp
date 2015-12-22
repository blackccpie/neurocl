#include "network.h"
#include "network_bnu.h"

int main()
{
    std::cout << "Welcome to neurocl!" << std::endl;

    try
    {
        neurocl::network_bnu net;
        //neurocl::network net;

        // Hardcoded topology
        std::vector<size_t> layer_sizes;
        layer_sizes.push_back( 16 ); // input L0
        layer_sizes.push_back( 12 ); // L1
        layer_sizes.push_back( 8 ); // L2
        layer_sizes.push_back( 4 ); // L3
        layer_sizes.push_back( 2 ); // L4
        layer_sizes.push_back( 1 ); // output L5
        net.add_layers_2d( layer_sizes );

        std::cout << "network populated OK" << std::endl;

        float test_sample[16*16];
        float test_output = 0.5f;

        for ( size_t i=0; i<16*16; i++ )
            test_sample[i] = std::rand()/float(RAND_MAX);

        net.set_training_sample( 16*16, test_sample, 1, &test_output );

        boost::chrono::system_clock::time_point start = boost::chrono::system_clock::now();

        std::cout << "set_input_sample OK" << std::endl;
        net.feed_forward();
        std::cout << "output = " << net.output() << std::endl;
        std::cout << "feed_forward OK" << std::endl;
        net.gradient_descent();
        std::cout << "gradient_descent OK" << std::endl;
        net.feed_forward();
        std::cout << "output = " << net.output() << std::endl;

        typedef boost::chrono::milliseconds bcms;
        bcms duration = boost::chrono::duration_cast<bcms>( boost::chrono::system_clock::now() - start );

        std::cout << "execution in "  << duration.count() << "ms"<< std::endl;

        std::cout << "Bye bye neurocl!" << std::endl;
    }
    catch( std::exception& e )
    {
        std::cerr << e.what() << std::endl;
    }
    return 0;
}
