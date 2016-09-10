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

#include "face_filer.h"

#include "common/samples_manager.h"
#include "common/network_exception.h"

#include "mlp/network_manager.h"

#include "facetools/edge_detect.h"
#include "facetools/face_detect.h"

#include <boost/lexical_cast.hpp>

#include <iostream>

using namespace cimg_library;

#define NEUROCL_EPOCH_SIZE 100
#define NEUROCL_BATCH_SIZE 10
#define MAX_MATCH_ERROR 0.1f

#define IMAGE_SIZEX 480
#define IMAGE_SIZEY 320

#define FACE_SIZEX 100
#define FACE_SIZEY 100

static const unsigned char green[] = { 0,255,0 };
static const unsigned char red[] = { 255,0,0 };

typedef enum
{
    FT_GUESS = 0,
    FT_USERA,
    FT_USERB,
    FT_UNKNOWN,
    //FT_NOT_A_FACE,
    FT_MAX
} face_type;

struct face_result
{
    face_result( face_type _type, float _score1, float _score2 )
        : type( _type ), score1( _score1 ), score2( _score2 ) {}

    const std::string result()
    {
        std::string str_type;
        switch( type )
        {
        case FT_USERA:
            str_type = "YOU ARE JOHN! ";
            break;
        case FT_USERB:
            str_type = "YOU ARE JANE! ";
            break;
        case FT_UNKNOWN:
        default:
            str_type = "YOU ARE UNKNOWN... ";
            break;
        }
        return str_type
            + "(" + boost::lexical_cast<std::string>( score1 ) + ";"
            + boost::lexical_cast<std::string>( score2 ) + ")";
    }

    face_type type;
    float score1;
    float score2;
};

boost::optional<face_result> opt_computed_face;

void face_preprocess( CImg<>& image )
{
    CImg<float> edged_image( 50, 50, 1, 1, 0 );

    sobel::process( image, edged_image );

    //canny<float> canny( image.width(), image.height() );
    //canny.process( image, edged_image );

    edged_image.normalize( 0.f, 1.f );
    image = edged_image; // overwrite input image
    //image.display();
}

void face_preprocess_generic( float* image, const size_t sizeX, const size_t sizeY )
{
    CImg<float> _image( image, sizeX, sizeY, 1, 1, true /*shared*/ );
    face_preprocess( _image );
}

void face_process(  CImg<float> image, const face_type& ftype,
                    neurocl::network_manager& net_manager,
                    neurocl::iterative_trainer& trainer,
                    face_filer& face_files )
{
    image.resize( 50, 50 );
    image.equalize( 256, 0, 255 );
    image.normalize( 0.f, 1.f );
    image.channel(0);

    CImg<float> work_image = image;

    face_preprocess( work_image );

    std::string label;
    float output[2] = { 0.f, 0.f };
    neurocl::sample sample( work_image.width() * work_image.height(), work_image.data(), 2, output );

    bool compute = false;

    switch( ftype )
    {
    case FT_USERA:
        label = "A";
        output[0] = 1.f;
        break;
    case FT_USERB:
        label = "B";
        output[1] = 1.f;
        break;
    case FT_UNKNOWN:
        label = "U";
        //output[2] = 1.f;
        break;
    //case FT_NOT_A_FACE:
    //    break;
    case FT_GUESS:
        compute = true;
        break;
    case FT_MAX:
    default:
        // should never be reached
    break;
    }

    if ( compute )
    {
        net_manager.compute_output( sample );

        std::cout << "max comp idx: " << sample.max_comp_idx() << " max comp val: " << sample.max_comp_val() << std::endl;

        if ( sample.max_comp_idx() == 0 )
            opt_computed_face = face_result( FT_USERA, output[0], output[1] );
        else if ( sample.max_comp_idx() == 1 )
            opt_computed_face = face_result( FT_USERB, output[0], output[1] );
    }
    else
    {
        trainer.train_new( sample );
        face_files.save_face( label, image );
    }
}

void grab_image( CImg<float>& image )
{
    int res = 0;
#ifdef __APPLE__
    // grab using ImageCapture utility
    res = system( "../../ImageCapture-v0.2/ImageCapture face_scene.png" );
#elif __arm__
    // grab using raspistill utility
    res = system( "raspistill -w 480 -h 320 -e png -o face_scene.png");
#else
    res = system( "fswebcam -r 480x320 --png -D face_scene.png");
#endif

    if ( res != -1 )
    {
    	image.load( "face_scene.png" );
    	image.resize( IMAGE_SIZEX, IMAGE_SIZEY );
    }
    else
        std::cerr << "error trying to grab webcam image!" << std::endl;
}

void draw_metadata( CImg<float>& image, const std::vector<face_detect::face_rect>& faces )
{
    std::string label( "Please center your face in the green rectangle and type:\nG = Guess?\nA = John\nE = Jane\nU = Unknown" );
    image.draw_text( 5, 5, label.c_str(), green );
    image.draw_rectangle( IMAGE_SIZEX/2 - FACE_SIZEX, IMAGE_SIZEY/2 - FACE_SIZEY,
        IMAGE_SIZEX/2 + FACE_SIZEX, IMAGE_SIZEY/2 + FACE_SIZEY, green, 1.f, ~0L );
    if ( !faces.empty() )
    {
        const face_detect::face_rect& frect = faces[0];
    	image.draw_rectangle( frect.x0, frect.y0, frect.x1, frect.y1, red, 1.f, ~0L );
    }
}

void draw_message( CImg<float>& image, const std::string& message )
{
    image.draw_text( IMAGE_SIZEX/2, IMAGE_SIZEY/2, message.c_str(), red );
}

int main ( int argc,char **argv )
{
    std::cout << "Welcome to facecam!" << std::endl;

    try
    {
        neurocl::network_manager net_manager( neurocl::network_manager::NEURAL_IMPL_BNU_REF );
        net_manager.load_network( "../nets/facecam/topology-facecam.txt", "../nets/facecam/weights-facecam.bin" );

        // TODO : check command arguments with boost
        // 0 normal use
        // 1 offline training
        bool offline_training = ( argc == 2 ) && ( boost::lexical_cast<int>( argv[1] ) == 1 );

        //************************* TRAINING *************************//

        if ( offline_training )
        {
            /******** TRAIN ********/

            neurocl::samples_manager& smp_manager = neurocl::samples_manager::instance();
            smp_manager.load_samples(  "../nets/facecam/facecam-train.txt",
                                                                true /*shuffle*/,
                                                                &face_preprocess_generic /* extra_preproc*/ );

            net_manager.batch_train( smp_manager, NEUROCL_EPOCH_SIZE, NEUROCL_BATCH_SIZE );

            /******** VALIDATE ********/

            const std::vector<neurocl::sample>& training_samples = smp_manager.get_samples();

            float mean_rmse = 0.f;
            size_t _rmse_score = 0;
            size_t _classif_score = 0;

            for ( size_t i = 0; i<training_samples.size(); i++ )
            {
                neurocl::test_sample tsample( smp_manager.get_samples()[i] );
                net_manager.compute_output( tsample );

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
        else
        {
            std::vector<face_detect::face_rect> faces;
            face_detect my_face_detect;

            face_filer face_files;

            neurocl::iterative_trainer trainer( net_manager, NEUROCL_BATCH_SIZE );

            CImg<float> input_image;
            CImg<float> display_image;

            CImgDisplay my_display( IMAGE_SIZEX, IMAGE_SIZEY );
            my_display.set_title( "FaceCam" );
        #ifdef __arm__
            my_display.set_fullscreen( true );
        #endif
            CImg<unsigned char> welcome( IMAGE_SIZEX, IMAGE_SIZEY, 1, 3 );
            welcome.draw_text( 50, 50, "Welcome to FaceCam\nPlease wait during capture initialization...", green );
            my_display.display( welcome );

            face_type ftype = FT_MAX;

            do
            {
                ftype = FT_MAX;

            	if ( my_display.is_key( cimg::keyG ) )
            	{
                    std::cout << "Guess that face!" << std::endl;
                    ftype = FT_GUESS;
                }
                else if ( my_display.is_key( cimg::keyA ) )
                {
                    std::cout << "This is John!" << std::endl;
                    ftype = FT_USERA;
                }
                else if ( my_display.is_key( cimg::keyE ) )
                {
                    std::cout << "This is Jane!" << std::endl;
                    ftype = FT_USERB;
                }
                else if ( my_display.is_key( cimg::keyU ) )
                {
                    std::cout << "This person is unknown!" << std::endl;
                    ftype = FT_UNKNOWN;
                }
                /*else if ( my_display.is_key( cimg::key0 ) )
                {
                    std::cout << "There is no one!" << std::endl;
                    ftype = FT_NOT_A_FACE;
                }*/
                else if ( my_display.is_key( cimg::keyQ ) || my_display.is_key( cimg::keyESC ) )
                {
                    std::cout << "Bye Bye!" << std::endl;
                    break;
                }
                else
                {
                    // UNMANAGED KEY
                    //std::cout << "unmanaged event" << std::endl;
                }

                std::cout << "key " << my_display.key() << std::endl;

                if ( ( ftype != FT_MAX ) && !faces.empty() )
                {
                    face_process(	input_image.get_crop( faces[0].x0, faces[0].y0, faces[0].x1, faces[0].y1 ), ftype,
                                    net_manager, trainer,
                                    face_files );
                }

                if ( opt_computed_face )
                {
                    std::cout << "face detected!" << std::endl;
                    draw_message( display_image, opt_computed_face.get().result() );
                }
                else
                {
                    grab_image( input_image );
                    display_image = input_image;
                    faces = my_face_detect.detect( input_image );
                    if ( faces.empty() )
                        draw_message( display_image, "NO FACE DETECTED!" );
                	draw_metadata( display_image, faces );
                }

                my_display.display( display_image );

                do
                {
                	my_display.wait();
                }
                while( my_display.key() == 0 );

                opt_computed_face = boost::none;

            } while( !my_display.is_closed() );

            net_manager.finalize_training_iteration();
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

    std::cout << "Bye bye facecam!" << std::endl;

    return 0;
}
