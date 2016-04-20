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

#include "thebrain.h"
#include "face_filer.h"
#include "chrono_manager.h"
#include "samples_manager.h"
#include "network_manager.h"
#include "network_exception.h"

#include "raspicam/raspicam.h"

#include "facetools/edge_detect.h"
#include "facetools/face_detect.h"

#include "CImg.h"

#include <boost/lexical_cast.hpp>

#include <iostream>
#include <ctime>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <sys/timeb.h>

using namespace cimg_library;

#define IMAGE_SIZEX 480
#define IMAGE_SIZEY 320

// EXPOSURE - raspicam::RASPICAM_EXPOSURE
/*
raspicam::RASPICAM_EXPOSURE_OFF;
raspicam::RASPICAM_EXPOSURE_AUTO;
raspicam::RASPICAM_EXPOSURE_NIGHT;
raspicam::RASPICAM_EXPOSURE_NIGHTPREVIEW;
raspicam::RASPICAM_EXPOSURE_BACKLIGHT;
raspicam::RASPICAM_EXPOSURE_SPOTLIGHT;
raspicam::RASPICAM_EXPOSURE_SPORTS;
raspicam::RASPICAM_EXPOSURE_SNOW;
raspicam::RASPICAM_EXPOSURE_BEACH;
raspicam::RASPICAM_EXPOSURE_VERYLONG;
raspicam::RASPICAM_EXPOSURE_FIXEDFPS;
raspicam::RASPICAM_EXPOSURE_ANTISHAKE;
raspicam::RASPICAM_EXPOSURE_FIREWORKS;
raspicam::RASPICAM_EXPOSURE_AUTO;
*/

// AUTO WHITE BALANCE - raspicam::RASPICAM_AWB
/*
raspicam::RASPICAM_AWB_OFF;
raspicam::RASPICAM_AWB_AUTO;
raspicam::RASPICAM_AWB_SUNLIGHT;
raspicam::RASPICAM_AWB_CLOUDY;
raspicam::RASPICAM_AWB_SHADE;
raspicam::RASPICAM_AWB_TUNGSTEN;
raspicam::RASPICAM_AWB_FLUORESCENT;
raspicam::RASPICAM_AWB_INCANDESCENT;
raspicam::RASPICAM_AWB_FLASH;
raspicam::RASPICAM_AWB_HORIZON;
raspicam::RASPICAM_AWB_AUTO;
*/

/*
cout<<"[-gr sets gray color mode]\n"<<endl;
cout<<"[-yuv sets yuv420 color mode]\n"<<endl;
cout<<"[-w width] [-h height] \n[-br brightness_val(0,100)]\n[-sh  sharpness_val (-100 to 100)]\n";
cout<<"[-co contrast_val (-100 to 100)]\n[-sa saturation_val (-100 to 100)]\n";
cout<<"[-iso ISO_val  (100 to 800)]\n[-vs turns on video stabilisation]\n[-ec exposure_compensation_value(-10,10)]\n";
cout<<"[-ss shutter_speed (value in microsecs (max 330000)]\n[-ec exposure_compensation_value(-10,10)]\n";
cout<<"[-exp mode (OFF,AUTO,NIGHT,NIGHTPREVIEW,BACKLIGHT,SPOTLIGHT,SPORTS,SNOW,BEACH,VERYLONG,FIXEDFPS,ANTISHAKE,FIREWORKS)]"<<endl;
cout<<"[-awb (OFF,AUTO,SUNLIGHT,CLOUDY,TUNGSTEN,FLUORESCENT,INCANDESCENT,FLASH,HORIZON)]"<<endl;
cout<<"[-nframes val: number of frames captured (100 default). 0 == Infinite lopp]\n";
cout<<"[-awb_r val:(0,8):set the value for the red component of white balance]"<<endl;
cout<<"[-awb_g val:(0,8):set the value for the green component of white balance]"<<endl;
*/

chrono_manager g_chrono;

#define MIN_FACE_RECO_SCORE 0.3f

static const unsigned char red[] = { 255,0,0 };
static const unsigned char green[] = { 0,255,0 };
static const unsigned char blue[] = { 0,0,255 };

struct face_result
{
    face_result( face_type _type, float _score1, float _score2 )
        : type( _type ), score1( _score1 ), score2( _score2 ) {}

    const std::string result( bool scores = false ) const
    {
        std::string str_type;
        switch( type )
        {
        case FT_USERA:
            str_type = "YOU ARE ALBERT! ";
            break;
        case FT_USERB:
            str_type = "YOU ARE ELSA! ";
            break;
        case FT_UNKNOWN:
        default:
            str_type = "YOU ARE UNKNOWN... ";
            break;
        }
        if ( scores )
			return str_type
            + "(" + boost::lexical_cast<std::string>( score1 ) + ";"
            + boost::lexical_cast<std::string>( score2 ) + ")";
		else
           return str_type;
    }

    const unsigned char* result_color() const
	{
		switch( type )
        {
        case FT_USERA:
            return green;
        case FT_USERB:
            return blue;
        case FT_UNKNOWN:
        default:
            return red;
        }
	}

    face_type type;
    float score1;
    float score2;
};

void face_preprocess( CImg<>& image )
{
    CImg<float> edged_image( 50, 50, 1, 1, 0 );

    sobel::process( image, edged_image );

    edged_image.normalize( 0.f, 1.f );
    image = edged_image; // overwrite input image
    //image.display();
}

void face_preprocess_generic( float* image, const size_t sizeX, const size_t sizeY )
{
    CImg<float> _image( image, sizeX, sizeY, 1, 1, true /*shared*/ );
    face_preprocess( _image );
}

const face_result face_process(  CImg<unsigned char> image, neurocl::network_manager& net_manager )
{
	CImg<float> work_image( image );

	g_chrono.step( "copying" );

    work_image.resize( 50, 50 );
    work_image.equalize( 256, 0, 255 );
    work_image.normalize( 0.f, 1.f );
    work_image.channel(0);

	g_chrono.step( "preparing" );

    face_preprocess( work_image );

    g_chrono.step( "preprocessing" );

    std::string label;
    float output[2] = { 0.f, 0.f };
    neurocl::sample sample( work_image.width() * work_image.height(), work_image.data(), 2, output );

	net_manager.compute_output( sample );

    g_chrono.step( "classification" );

	//std::cout << "max comp idx: " << sample.max_comp_idx() << " max comp val: " << sample.max_comp_val() << std::endl;

	if (sample.max_comp_val() < MIN_FACE_RECO_SCORE )
		return face_result( FT_UNKNOWN, output[0], output[1] );
	else if ( sample.max_comp_idx() == 0 )
		return face_result( FT_USERA, output[0], output[1] );
	else if ( sample.max_comp_idx() == 1 )
		return face_result( FT_USERB, output[0], output[1] );
}

void draw_metadata( CImg<unsigned char>& image, const std::vector<face_detect::face_rect>& faces, const face_result& fresult )
{
    if ( !faces.empty() )
    {
        const face_detect::face_rect& frect = faces[0];
    	image.draw_rectangle( frect.x0, frect.y0, frect.x1, frect.y1, fresult.result_color(), 1.f, ~0L );
    	image.draw_text( frect.x0, frect.y0-20, fresult.result().c_str(), fresult.result_color() );
    }
}

void draw_message( CImg<unsigned char>& image, const std::string& message )
{
    image.draw_text( IMAGE_SIZEX/2, IMAGE_SIZEY/2, message.c_str(), red );
}

void draw_fps( CImg<unsigned char>& image, const float& fps )
{
	std::stringstream ss;
	ss << std::setprecision(1) << fps << "FPS";
    image.draw_text( 15, 15, ss.str().c_str(), red );
}

bool _is_valid_face( face_detect::face_rect face )
{
	int face_width = face.x1 - face.x0;
	int face_height = face.y1 - face.y0;
	return ( face_width > 100 ) && ( face_width < 250 ) 
		&& ( face_height > 100 ) && ( face_height < 250 );
}

void _main_train( raspicam::RaspiCam& camera, cimg_library::CImgDisplay& my_diplay );
void _main_live( raspicam::RaspiCam& camera, cimg_library::CImgDisplay& my_diplay );

int main ( int argc,char **argv )
{
    std::cout << "Welcome to neuropicam!" << std::endl;

	raspicam::RaspiCam camera;

	try
	{
        camera.setWidth( IMAGE_SIZEX );
        camera.setHeight( IMAGE_SIZEY );
        camera.setBrightness( 50 );
        camera.setSharpness( 0 );
        camera.setContrast( 0 );
        camera.setSaturation( 0 );
        camera.setShutterSpeed( 0 );
        camera.setISO( 400 );
        //camer3.setVideoStabilization( true );
        camera.setExposureCompensation( 0 );
        //camera.setFormat(raspicam::RASPICAM_FORMAT_GRAY);
        camera.setFormat(raspicam::RASPICAM_FORMAT_RGB);
        //camera.setExposure( /**/ );
        //camera.setAWB( /**/ );
        camera.setAWB_RB( 1, 1 );

        std::cout << "Connecting to camera" << std::endl;

        if ( !camera.open() )
        {
            std::cerr << "Error opening camera" << std::endl;
            return -1;
        }
        std::cout << "Connected to camera =" << camera.getId() << " bufs=" << camera.getImageBufferSize() << std::endl;

        cimg_library::CImgDisplay my_display( IMAGE_SIZEX, IMAGE_SIZEY );
        my_display.set_title( "NeuroPiCam" );
#ifdef __arm__
        my_display.set_fullscreen( true );
#endif

        bool auto_training = ( argc == 2 ) && ( boost::lexical_cast<int>( argv[1] ) == 1 );

        if ( auto_training )
        {
            _main_train( camera, my_display );
        }
        else
            _main_live( camera, my_display );
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

    std::cout << "Bye bye neuropicam!" << std::endl;

    camera.release();

    return 0;
}

void _main_train( raspicam::RaspiCam& camera, cimg_library::CImgDisplay& my_display )
{
	std::vector<face_detect::face_rect> faces;
    face_detect my_face_detect;
	
    // TODO : remove/backup existing weights

    neurocl::network_manager net_manager( neurocl::network_manager::NEURAL_IMPL_BNU );
    net_manager.load_network( "../nets/facecam/topology-facecam.txt", "../nets/facecam/weights-facecam.bin" );

	// TODO : remove existing training file
	std::ofstream auto_train_file( "../nets/facecam/auto-train.txt" );

    boost::shared_array<unsigned char> data( new unsigned char[ camera.getImageBufferSize() ] );

    std::cout << "Capturing...." << std::endl;

	CImg<unsigned char> display_image;

	face_filer face_files;

	size_t user_faces = 0;

    do
    {
        // CAPTURE USER A
        camera.grab();
        camera.retrieve( data.get() );
        
        cimg_library::CImg<unsigned char> input_image( data.get(), IMAGE_SIZEX, IMAGE_SIZEY, 1, 1, true );

        display_image = input_image;

        faces = my_face_detect.detect( input_image );

        bool valid_face = !faces.empty() && _is_valid_face( faces[0] );
			
		if ( valid_face )
        {
			CImg<float> work_image( input_image );
			
			work_image.resize( 50, 50 );
			work_image.equalize( 256, 0, 255 );
			work_image.normalize( 0.f, 1.f );
			work_image.channel(0);

			face_preprocess( work_image );
			
			face_files.save_face( "autoA", work_image );
			
			auto_train_file << face_files.last_path() << " 1 0" << std::endl;
			
            //draw_metadata( display_image, faces, fres );

			++user_faces;
        }
    } while( user_faces < 20 );

	user_faces = 0;

    // CHANGE USER

    do
    {
        // CAPTURE USER B
        camera.grab();
        camera.retrieve( data.get() );
        
		cimg_library::CImg<unsigned char> input_image( data.get(), IMAGE_SIZEX, IMAGE_SIZEY, 1, 1, true );

        display_image = input_image;

        faces = my_face_detect.detect( input_image );

        bool valid_face = !faces.empty() && _is_valid_face( faces[0] );
			
		if ( valid_face )
        {
			CImg<float> work_image( input_image );
			
			work_image.resize( 50, 50 );
			work_image.equalize( 256, 0, 255 );
			work_image.normalize( 0.f, 1.f );
			work_image.channel(0);

			face_preprocess( work_image );
			
			face_files.save_face( "autoB", work_image );
			
			auto_train_file << face_files.last_path() << " 1 0" << std::endl;
			
            //draw_metadata( display_image, faces, fres );

			++user_faces;
        }
    } while( user_faces < 20 );

	// ADD UNKNOWN USER
	for ( int i=0; i<20; i++ )
		auto_train_file << "../nets/facecam/faces/autoU/" << i << ".png 0 1" << std::endl;

	auto_train_file.close();

    // TRAIN THE WHOLE NETWORK
    
	neurocl::samples_manager& smp_manager = neurocl::samples_manager::instance();
	smp_manager.load_samples(  "../nets/facecam/auto-train.txt",
														true /*shuffle*/,
														&face_preprocess_generic /* extra_preproc*/ );

	net_manager.batch_train( smp_manager, 500 /*epoch*/, 20 /*batch*/ );
}

void _main_live( raspicam::RaspiCam& camera, cimg_library::CImgDisplay& my_display )
{
    thebrain my_brain;

    std::vector<face_detect::face_rect> faces;
    face_detect my_face_detect;

    neurocl::network_manager net_manager( neurocl::network_manager::NEURAL_IMPL_BNU );
    net_manager.load_network( "../nets/facecam/topology-facecam.txt", "../nets/facecam/weights-facecam.bin" );

    boost::shared_array<unsigned char> data( new unsigned char[ camera.getImageBufferSize() ] );

    std::cout << "Capturing.... (buffer size : " << camera.getImageBufferSize() << ")" << std::endl;

    CImg<unsigned char> display_image;
	cimg_library::CImg<unsigned char> input_image( IMAGE_SIZEX, IMAGE_SIZEY, 1, 3, true );

    size_t i=0;

    do
    {
        if ( my_display.is_key( cimg::keyQ ) || my_display.is_key( cimg::keyESC ) )
        {
            std::cout << "Bye Bye!" << std::endl;
            break;
        }

        g_chrono.start();

        camera.grab();
        camera.retrieve( data.get() );

        g_chrono.step( "grabbing" );

        //cimg_library::CImg<unsigned char> input_image( data.get(), IMAGE_SIZEX, IMAGE_SIZEY, 1, 1, true );

		// RGB deinterlacing
		cimg_forXYC(input_image,x,y,v) { input_image(x,y,v) = data[3*(x+(y*IMAGE_SIZEX))+v]; }

        display_image = input_image;

        faces = my_face_detect.detect( input_image );

        bool valid_face = !faces.empty() && _is_valid_face( faces[0] );
			
		if ( !valid_face )
            draw_message( display_image, "NO FACE DETECTED!" );
        else
        {
            const face_result fres = face_process( input_image.get_crop( faces[0].x0, faces[0].y0, faces[0].x1, faces[0].y1 ), net_manager );
            draw_metadata( display_image, faces, fres );

            my_brain.push_face_type( fres.type );
        }

        std::cout << g_chrono.summary() << std::endl;

        g_chrono.frame();
        draw_fps( display_image, g_chrono.framerate() );

        my_display.display( display_image );

    } while(true);
}
