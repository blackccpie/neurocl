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

#include "neurocl.h"

#include "raspicam/raspicam.h"

#include "imagetools/edge_detect.h"
#include "imagetools/face_detect.h"

#include "CImg.h"

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>

#include <fstream>

using namespace neurocl;
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

#define MIN_FACE_RECO_SCORE 0.5f

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
        case face_type::FT_USERA:
            str_type = "YOU ARE " + boost::to_upper_copy( facecam_users::instance().nicknameA() ) + "! ";
            break;
        case face_type::FT_USERB:
            str_type = "YOU ARE " + boost::to_upper_copy( facecam_users::instance().nicknameB() ) + "! ";
            break;
        case face_type::FT_UNKNOWN:
        default:
            str_type = "YOU ARE UNKNOWN... ";
            break;
        }
        if ( scores )
			return str_type
            + "(" + std::to_string( score1 ) + ";"
            + std::to_string( score2 ) + ")";
		else
           return str_type;
    }

    const unsigned char* result_color() const
	{
		switch( type )
        {
        case face_type::FT_USERA:
            return green;
        case face_type::FT_USERB:
            return blue;
        case face_type::FT_UNKNOWN:
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

const face_result face_process(  CImg<unsigned char> image, std::shared_ptr<network_manager_interface> net_manager )
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
    sample sample( work_image.width() * work_image.height(), work_image.data(), 2, output );

	net_manager->compute_output( sample );

    g_chrono.step( "classification" );

	//std::cout << "max comp idx: " << sample.max_comp_idx() << " max comp val: " << sample.max_comp_val() << std::endl;

	if (sample.max_comp_val() < MIN_FACE_RECO_SCORE )
		return face_result( face_type::FT_UNKNOWN, output[0], output[1] );
	else if ( sample.max_comp_idx() == 0 )
		return face_result( face_type::FT_USERA, output[0], output[1] );
	else if ( sample.max_comp_idx() == 1 )
		return face_result( face_type::FT_USERB, output[0], output[1] );
    else
    {
        std::cout << "Warning : unmanaged use case" << std::endl;
        return face_result( face_type::FT_UNKNOWN, output[0], output[1] );
    }
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

void draw_metadata( CImg<unsigned char>& image, const std::vector<face_detect::face_rect>& faces, const std::string& label )
{
    if ( !faces.empty() )
    {
		const face_detect::face_rect& frect = faces[0];
    	image.draw_rectangle( frect.x0, frect.y0, frect.x1, frect.y1, green, 1.f, ~0L );
    	image.draw_text( frect.x0, frect.y0-20, label.c_str(), green );
	}
}

void draw_message( CImg<unsigned char>& image, const std::string& message, const int xorig = IMAGE_SIZEX/2 )
{
    image.draw_text( xorig, IMAGE_SIZEY/2, message.c_str(), red );
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
void _main_live( raspicam::RaspiCam& camera, cimg_library::CImgDisplay& my_diplay, bool auto_trained );

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
        bool auto_trained = ( argc == 2 ) && ( boost::lexical_cast<int>( argv[1] ) == 2 );

        if ( auto_training )
        {
            _main_train( camera, my_display );
        }
        else
            _main_live( camera, my_display, auto_trained );
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

    std::cout << "Bye bye neuropicam!" << std::endl;

    camera.release();

    return 0;
}

#define NB_TRAINING_FACES 20

const std::string g_weights_facecam_auto = "../nets/facecam/weights-facecam-auto.bin";
const std::string g_training_file_auto = "../nets/facecam/auto-train.txt";

void progress( int percent, cimg_library::CImgDisplay& my_display, cimg_library::CImg<unsigned char>& display_image )
{
	display_image = (unsigned char)0;
	draw_message( display_image, "NEURAL NETWORK TRAINING PROGRESS : " + std::to_string( percent ) + "%", 20 );
	my_display.display( display_image );
}

void _main_train( raspicam::RaspiCam& camera, cimg_library::CImgDisplay& my_display )
{
	using namespace boost::filesystem;

	// TODO : define in face_commons?
    // constexpr compile error due to non-litteral types
    const/*expr*/ std::array<std::string,2> users {"autoA","autoB"};
    const/*expr*/ std::array<std::string,2> scores {"1 0","0 1"};


	std::vector<face_detect::face_rect> faces;
    face_detect my_face_detect;

    // remove/backup existing auto weights
    if ( exists( g_weights_facecam_auto ) )
	{
		copy_file( g_weights_facecam_auto, g_weights_facecam_auto + ".save", copy_option::overwrite_if_exists );
		remove( g_weights_facecam_auto );
	}

    std::shared_ptr<network_manager_interface> net_manager = network_factory::build( network_factory::t_neural_impl::NEURAL_IMPL_MLP );
    net_manager->load_network( "../nets/facecam/topology-facecam.txt", g_weights_facecam_auto );

	// remove existing training file + image files
	if ( exists( g_training_file_auto ) )
		remove( g_training_file_auto );
	for ( const std::string& user : users )
	{
		if ( exists( "/home/pi/Pictures/facecam_faces/" + user ) )
			remove_all( "/home/pi/Pictures/facecam_faces/" + user );
	}

	// create new training file
	std::ofstream auto_train_file( g_training_file_auto.c_str() );

    boost::shared_array<unsigned char> data( new unsigned char[ camera.getImageBufferSize() ] );

    std::cout << "Capturing...." << std::endl;

	CImg<unsigned char> display_image( IMAGE_SIZEX, IMAGE_SIZEY, 1, 3 );
	cimg_library::CImg<unsigned char> input_image( IMAGE_SIZEX, IMAGE_SIZEY, 1, 3, true );

	face_filer face_files;

	for ( size_t u=0; u<users.size(); u++ )
	{
		size_t user_faces = 0;

		display_image = (unsigned char)0;
		draw_message( display_image, "PRESS A KEY WHEN READY TO CAPTURE USER : " + users[u], 10 );

		my_display.display( display_image );

		do
		{
			my_display.wait();
		}
		while( my_display.key() == 0 );

		do
		{
			// CAPTURE USER
			camera.grab();
			camera.retrieve( data.get() );

			//cimg_library::CImg<unsigned char> input_image( data.get(), IMAGE_SIZEX, IMAGE_SIZEY, 1, 1, true );

			// RGB deinterlacing
			cimg_forXYC(input_image,x,y,v) { input_image(x,y,v) = data[3*(x+(y*IMAGE_SIZEX))+v]; }

			display_image = input_image;

			faces = my_face_detect.detect( input_image.data(), input_image.width(), input_image.height() );

			bool valid_face = !faces.empty() && _is_valid_face( faces[0] );

			if ( valid_face )
			{
				CImg<float> work_image( input_image.get_crop( faces[0].x0, faces[0].y0, faces[0].x1, faces[0].y1 ) );

				work_image.resize( 50, 50 );
				work_image.equalize( 256, 0, 255 );
				work_image.normalize( 0.f, 1.f );
				work_image.channel(0);

				face_files.save_face( users[u], work_image );

				auto_train_file << face_files.last_path() << " " << scores[u] << std::endl;

				draw_metadata( display_image, faces, users[u] + " - " + std::to_string( user_faces+1 ) );

				my_display.display( display_image );

				++user_faces;
			}
		} while( user_faces < NB_TRAINING_FACES );
	}

	// ADD UNKNOWN USER
	for ( int i=0; i<20; i++ )
		auto_train_file << "/home/pi/Pictures/facecam_faces/autoU/" << i << ".png 0 0" << std::endl;

	auto_train_file.close();

	camera.release();

    // TRAIN THE WHOLE NETWORK
	neurocl::samples_manager smp_manager;
	smp_manager.load_samples(  "../nets/facecam/auto-train.txt",
														true /*shuffle*/,
														&face_preprocess_generic /* extra_preproc*/ );

	net_manager->batch_train( 	smp_manager,
								100 /*epoch*/,
								20 /*batch*/,
								std::bind( &progress, std::placeholders::_1, my_display, display_image ) );
}

void _main_live( raspicam::RaspiCam& camera, cimg_library::CImgDisplay& my_display, bool auto_trained )
{
    thebrain my_brain;

    std::vector<face_detect::face_rect> faces;
    face_detect my_face_detect;

    std::shared_ptr<network_manager_interface> net_manager = network_factory::build( network_factory::t_neural_impl::NEURAL_IMPL_MLP );
    if ( !auto_trained )
		net_manager->load_network( "../nets/facecam/topology-facecam.txt", "../nets/facecam/weights-facecam.bin" );
	else
		net_manager->load_network( "../nets/facecam/topology-facecam.txt", g_weights_facecam_auto );

    boost::shared_array<unsigned char> data( new unsigned char[ camera.getImageBufferSize() ] );

    std::cout << "Capturing.... (buffer size : " << camera.getImageBufferSize() << ")" << std::endl;

    CImg<unsigned char> display_image;
	cimg_library::CImg<unsigned char> input_image( IMAGE_SIZEX, IMAGE_SIZEY, 1, 3, true );

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

        faces = my_face_detect.detect( input_image.data(), input_image.width(), input_image.height() );

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
