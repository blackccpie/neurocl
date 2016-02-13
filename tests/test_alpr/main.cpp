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

#include "alphanum.h"

#include "network_manager.h"
#include "network_exception.h"
#include "samples_manager.h"

#include "CImg.h"

#include <boost/lexical_cast.hpp>

#include <iostream>

#define NEUROCL_EPOCH_SIZE 10
#define NEUROCL_BATCH_SIZE 10
#define MAX_MATCH_ERROR 0.1f

using namespace cimg_library;

int main( int argc, char *argv[] )
{
    std::cout << "Welcome to test_alpr!" << std::endl;

    try
    {
        // TODO : check command arguments with boost

        bool training_enabled = boost::lexical_cast<bool>( argv[4] );

        neurocl::network_manager net_manager( neurocl::network_manager::NEURAL_IMPL_BNU );
        net_manager.load_network( argv[2], argv[3] );

        //************************* TRAINING *************************//

        if ( training_enabled )
        {
            /******** TRAIN ********/

            neurocl::samples_manager& smp_manager = neurocl::samples_manager::instance();
            neurocl::samples_manager::instance().load_samples( argv[1] );

            net_manager.batch_train( smp_manager, NEUROCL_EPOCH_SIZE, NEUROCL_BATCH_SIZE );

            /******** VALIDATE ********/

            const std::vector<neurocl::sample>& training_samples = smp_manager.get_samples();

            size_t _rmse_score = 0;
            size_t _classif_score = 0;

            for ( size_t i = 0; i<training_samples.size(); i++ )
            {
                neurocl::test_sample tsample( smp_manager.get_samples()[i] );
                net_manager.compute_output( tsample );

                std::cout << tsample.output() << std::endl;
                std::cout << tsample.ref_output() << std::endl;
                std::cout << tsample.RMSE() << std::endl;

                if ( tsample.RMSE() < MAX_MATCH_ERROR )
                    ++ _rmse_score;

                if ( tsample.classified() )
                    ++_classif_score;

            	//std::cout << "TEST OUTPUT IS : " << tsample.output() << std::endl;
            }

            std::cout << "RMSE SCORE IS " << _rmse_score << "/" << training_samples.size() << std::endl;
            std::cout << "CLASSIF SCORE IS " << _classif_score << "/" << training_samples.size() << std::endl;
        }

        //************************* TESTING *************************//

        else
        {
            const size_t sizeX = 50;
            const size_t sizeY = 100;

            boost::shared_array<float> output( new float[36] );

            CImg<float> input_plate( argv[1] );

            // Compute rduced plate image
            CImg<float> reduced_plate =
                input_plate.resize( sizeY * input_plate.width() / input_plate.height(), sizeY );
            input_plate.clear();

            reduced_plate.equalize( 256, 0, 255 );
            reduced_plate.normalize( 0.f, 1.f );
            reduced_plate.channel(0);
            reduced_plate = 1.f - reduced_plate;

            // Remove a 10px border
            cimg_for_borderXY( reduced_plate, x, y, 10 ) { reduced_plate( x, y ) = 0; }

            reduced_plate.display();

            // Compute row sums image
            CImg<float> row_sums( reduced_plate.width(), 1 );
            cimg_forX( row_sums, x )
            {
                float rsum = 0.f;
                cimg_forY( reduced_plate, y )
                {
                    rsum += reduced_plate(x,y);
                }
                row_sums(x) = rsum;
            }

            // Display row sums graph
            CImg<float> graph( reduced_plate.width(), 400, 1, 3, 0 );
            unsigned char green[] = { 0,255,0 };
            graph.draw_graph( row_sums, green, 1, 1, 1, 0, 50 );
            graph.display();

            // Initialize distance map
            CImg<float> dist_map( reduced_plate.width(), 1, 1, 1, 0 );

            //size_t i = 51;
            for ( size_t i=0; i<=( reduced_plate.width() - sizeX ); i++ )
            {
                CImg<float> subimage = reduced_plate.get_columns( i, i + sizeX - 1 );
                cimg_for_borderXY( subimage, x, y, 2 ) { subimage( x, y ) = 0; }
                subimage.threshold( 0.5f );
                //subimage.erode( 3 );

                neurocl::sample sample( sizeX * sizeY, subimage.data(), 36, output.get() );

                net_manager.compute_output( sample );

                dist_map(i) = sample.max_comp_val();

                //std::cout << sample.output() << std::endl;

                if ( sample.max_comp_val() > 0.8f )
                {
                    CImg<float> disp_image( 50, 100, 1, 3 );
                    cimg_forXYC( disp_image, x, y, c ) {
                        disp_image( x, y, c ) = 255.f * subimage( x, y ); }

                    unsigned char green[] = { 0,255,0 };
                    std::string label = alphanum( sample.max_comp_idx() ).string() + " "
                        + boost::lexical_cast<std::string>( sample.max_comp_val() );
                    disp_image.draw_text( 5, 5, label.c_str(), green );
                    disp_image.display();
                }

                //subimage.normalize(0, 255);
                //subimage.save( "sample-At.png" );

            	std::cout << "TEST OUTPUT IS : " << sample.output() << std::endl;
            }
        }
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

    std::cout << "Bye bye test_alpr!" << std::endl;

    return 0;
}
