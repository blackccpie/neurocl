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
#include "network_exception.h"

#include "CImg.h"

#include <boost/foreach.hpp>
#include <boost/filesystem.hpp>
namespace bfs = boost::filesystem;

#include <iostream>

// return the filenames of all files that have the specified extension
// in the specified directory and all subdirectories
void get_all( const bfs::path& root, const std::string& ext, std::vector<bfs::path>& ret )
{
    if( !bfs::exists( root ) || !bfs::is_directory( root ) )
        return;

    bfs::recursive_directory_iterator it( root );
    bfs::recursive_directory_iterator endit;

    while( it != endit )
    {
        if( bfs::is_regular_file(*it) && it->path().extension() == ext )
            ret.push_back( it->path() ) ;
        ++it;
    }
}

cimg_library::CImg<float> get_preprocessed_image( const std::string& file )
{
    cimg_library::CImg<float> img( file.c_str() );

    //const int new_size = std::max<int>( img.width(), img.height() );
    //img.resize( new_size, new_size );
    img.equalize( 256, 0, 255 );
    img.normalize( 0.f, 1.f );
    img.channel(0);
    //img.display();
    return img;
    //return img.resize( 64, 64 );
}

int main( int argc, char *argv[] )
{
    std::cout << "Welcome to neurocl!" << std::endl;

    try
    {
        neurocl::network_manager net_manager( neurocl::network_manager::NEURAL_IMPL_BNU );

        net_manager.load_network( "titi" );

        const bfs::path input_positive_path( argv[1] );
        const std::string input_positive_ext( argv[2] );
        const bfs::path input_negative_path( argv[3] );
        const std::string input_negative_ext( argv[4] );

        std::vector<bfs::path> input_positive_samples_paths;
        get_all( input_positive_path, input_positive_ext, input_positive_samples_paths );

        std::cout << input_positive_samples_paths.size() << " positive sample files have been listed for training" << std::endl;

        std::vector<bfs::path> input_negative_samples_paths;
        get_all( input_negative_path, input_negative_ext, input_negative_samples_paths );

        std::cout << input_negative_samples_paths.size() << " negative sample files have been listed for training" << std::endl;

        // output for face recon is expected to 1
        float face_output = 1.0f;
        float non_face_output = 0.0f;

        //************************* TRAINING *************************//

        std::vector<bfs::path> rand_input_samples_paths;
        rand_input_samples_paths.reserve( input_positive_samples_paths.size() + input_negative_samples_paths.size() ); // preallocate memory
        rand_input_samples_paths.insert( rand_input_samples_paths.end(), input_positive_samples_paths.begin(), input_positive_samples_paths.end() );
        rand_input_samples_paths.insert( rand_input_samples_paths.end(), input_negative_samples_paths.begin(), input_negative_samples_paths.end() );

        for ( int i=0; i<30; i++ )
        {
            std::random_shuffle ( rand_input_samples_paths.begin(), rand_input_samples_paths.end() );

            net_manager.prepare_training_iteration();

            BOOST_FOREACH( const bfs::path& sample_path, rand_input_samples_paths )
            {
                const std::string sample_file = sample_path.string();
                std::cout << "loading image : " << sample_file << std::endl;

                cimg_library::CImg<float> img = get_preprocessed_image( sample_file );

                std::cout << "image loaded"  << std::endl;

                float* sample_output = 0;
                if ( sample_file.find( input_positive_path.string() ) != std::string::npos )
                    sample_output = &face_output;
                else
                    sample_output = &non_face_output;

                net_manager.train(
                    neurocl::sample( img.size(), img.data(), 1, sample_output ) );

                // Dump weights for debugging purposes
                //net_manager.dump_weights();
                //net_manager.dump_activations();
            }

            net_manager.finalize_training_iteration();
        }

        /*
        // POSITIVE TRAINING
        BOOST_FOREACH( const bfs::path& sample_path, input_positive_samples_paths )
        {
            const std::string sample_file = sample_path.string();
            std::cout << "loading positive image : " << sample_file << std::endl;

            cimg_library::CImg<float> img = get_preprocessed_image( sample_file );

            net_manager.train(
                neurocl::sample( img.size(), img.data(), 1, &face_output ) );

            static int pos_stop = 0;
            if ( pos_stop++ == 50 )
                break;
        }

        // NEGATIVE TRAINING
        BOOST_FOREACH( const bfs::path& sample_path, input_negative_samples_paths )
        {
            const std::string sample_file = sample_path.string();
            std::cout << "loading negative image : " << sample_file << std::endl;

            cimg_library::CImg<float> img = get_preprocessed_image( sample_file );

            net_manager.train(
                neurocl::sample( img.size(), img.data(), 1, &non_face_output ) );

            static int neg_stop = 0;
            if ( neg_stop++ == 50 )
                break;
        }*/

        // Dump weights for debugging purposes
        //net_manager.dump_weights();

        //************************* AFTER TESTING *************************//

        float test_output = -1.f;
        {
            cimg_library::CImg<float> non_face_img = get_preprocessed_image( input_negative_samples_paths[0].string() );
            neurocl::sample non_face_sample( non_face_img.size(), non_face_img.data(), 1, &test_output );
            net_manager.compute_output( non_face_sample );
        }

        std::cout << "NON FACE OUTPUT SCORE IS : " << test_output << std::endl;

        test_output = -1.f;
        {
            cimg_library::CImg<float> face_img = get_preprocessed_image( input_positive_samples_paths[0].string() );
            neurocl::sample face_sample( face_img.size(), face_img.data(), 1, &test_output );
            net_manager.compute_output( face_sample );
        }

        std::cout << "FACE OUTPUT SCORE IS : " << test_output << std::endl;

// FORMER TEST CODE
/*
        float test_sample[64*64];
        float test_output = 0.5f;

        for ( size_t i=0; i<64*64; i++ )
            test_sample[i] = std::rand()/float(RAND_MAX);

        std::vector<neurocl::sample> training_set;
        training_set.push_back( neurocl::sample( 64*64, test_sample, 1, &test_output ) );

        net_manager.train( training_set );
*/

        std::cout << "Bye bye neurocl!" << std::endl;
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
    return 0;
}
