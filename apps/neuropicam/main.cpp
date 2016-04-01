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

#include "raspicam.h"

#include "CImg.h"

#include <iostream>
#include <ctime>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <sys/timeb.h>

using namespace std;

#define IMAGE_SIZEX 480
#define IMAGE_SIZEY 320

size_t nFramesCaptured=1000;

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

//timer functions
#include <sys/time.h>
#include <unistd.h>
class Timer{
    private:
    struct timeval _start, _end;

public:
    Timer(){}
    void start(){
        gettimeofday(&_start, NULL);
    }
    void end(){
        gettimeofday(&_end, NULL);
    }
    double getSecs(){
    return double(((_end.tv_sec  - _start.tv_sec) * 1000 + (_end.tv_usec - _start.tv_usec)/1000.0) + 0.5)/1000.;
    }

};

int main ( int argc,char **argv )
{
	raspicam::RaspiCam camera;

	camera.setWidth( IMAGE_SIZEX );
	camera.setHeight( IMAGE_SIZEY );
	camera.setBrightness( 50 );
	camera.setSharpness( 0 );
	camera.setContrast( 0 );
	camera.setSaturation( 0 );
	camera.setShutterSpeed( 0 );
	camera.setISO( 400 );
	//camera.setVideoStabilization( true );
	camera.setExposureCompensation( 0 );
	camera.setFormat(raspicam::RASPICAM_FORMAT_GRAY);
	//camera.setFormat(raspicam::RASPICAM_FORMAT_YUV420);
	//camera.setExposure( /**/ );
	//camera.setAWB( /**/ );
	camera.setAWB_RB( 1, 1 );

	cout << "Connecting to camera" << endl;

    if ( !camera.open() )
    {
        cerr << "Error opening camera" << endl;
        return -1;
    }
    cout << "Connected to camera =" << camera.getId() << " bufs=" << camera.getImageBufferSize() << endl;
    unsigned char *data=new unsigned char[  camera.getImageBufferSize( )];
    Timer timer;

	cimg_library::CImgDisplay my_display( IMAGE_SIZEX, IMAGE_SIZEY );
	my_display.set_title( "NeuroPiCam" );
	my_display.set_fullscreen( true );

    cout << "Capturing...." << endl;
    
    size_t i=0;
    
    timer.start();
    
	do
    {
        camera.grab();
        camera.retrieve ( data );

        cimg_library::CImg<unsigned char> img( data, IMAGE_SIZEX, IMAGE_SIZEY, 1, 1 );
        
        my_display.display( img );

		if ( i%5==0 )
		{
			cout << "\r capturing ..." << i << "/" << nFramesCaptured << std::flush;
		}

    } while(++i<nFramesCaptured || nFramesCaptured==0); //stops when nFrames captured or at infinity lpif nFramesCaptured<0

    timer.end();

    cerr<< timer.getSecs()<< " seconds for "<< nFramesCaptured<< "  frames : FPS " << ( ( float ) ( nFramesCaptured ) / timer.getSecs() ) <<endl;

    camera.release();

    return 0;
}
