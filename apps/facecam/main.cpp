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

#include "CImg.h"

#include <iostream>

//#include <stdlib.h>

using namespace cimg_library;

typedef enum
{
    FT_GUESS = 0,
    FT_ALBERT,
    FT_ELSA,
    FT_UNKNOWN,
    FT_NOT_A_FACE,
    FT_MAX
} face_type;

void process( CImg<float> image, const face_type& ftype, neurocl::network_manager& net_manager )
{
    float output[3] = { 0.f, 0.f, 0.f };
    neurocl::sample sample( image.width() * image.height(), image.data(), 3, output );

    bool compute = false;

    switch( ftype )
    {
    case FT_ALBERT:
        output[0] = 1.f;
        break;
    case FT_ELSA:
        output[1] = 1.f;
        break;
    case FT_UNKNOWN:
        output[2] = 1.f;
        break;
    case FT_NOT_A_FACE:
        break;
    case FT_GUESS:
        compute = true;
        break;
    case FT_MAX:
    default:
        // should never be reached
    break;
    }

    if ( compute )
        net_manager.compute_output( sample );
    else
        net_manager.train( sample );
}

#define IMAGE_SIZEX 480
#define IMAGE_SIZEY 320

#define FACE_SIZEX 80
#define FACE_SIZEY 100

void grab_image( CImg<float>& image )
{
#ifdef __APPLE__
    // grab using ImageCapture utility
    system( "../../ImageCapture-v0.2/ImageCapture face_scene.png" );
#else
    // grab using raspistill utility
    system( "raspistill -w 480 -h 320 -o face_scene.png");
#endif
    image.load( "face_scene.png" );
    image.resize( IMAGE_SIZEX, IMAGE_SIZEY );

    unsigned char green[] = { 0,255,0 };
    std::string label( "Please center your face in the green rectangle and type:\nG = Guess?\nA = Albert\nE = Elsa\nU = Unknown\n0 = Not a face!" );
    image.draw_text( 5, 5, label.c_str(), green );
    image.draw_rectangle( IMAGE_SIZEX/2 - FACE_SIZEX, IMAGE_SIZEY/2 - FACE_SIZEY,
        IMAGE_SIZEX/2 + FACE_SIZEX, IMAGE_SIZEY/2 + FACE_SIZEY, green, 1.f, ~0L );
}

int main ( int argc,char **argv )
{
    neurocl::network_manager net_manager( neurocl::network_manager::NEURAL_IMPL_BNU );
    //net_manager.load_network( "../nets/facecam/topology-facecam.txt", "../nets/facecam/weights-facecam.txt" );

    CImg<float> input_image;

    CImgDisplay my_display;

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
            std::cout << "This is Albert!" << std::endl;
            ftype = FT_ALBERT;
        }
        else if ( my_display.is_key( cimg::keyE ) )
        {
            std::cout << "This is Elsa!" << std::endl;
            ftype = FT_ELSA;
        }
        else if ( my_display.is_key( cimg::keyU ) )
        {
            std::cout << "This person is unknown!" << std::endl;
            ftype = FT_UNKNOWN;
        }
        else if ( my_display.is_key( cimg::key0 ) )
        {
            std::cout << "There is no one!" << std::endl;
            ftype = FT_NOT_A_FACE;
        }

        if ( ftype != FT_MAX )
        {
            process( input_image.crop( 100, 100, 200, 200 ), ftype, net_manager );

            grab_image( input_image );
            my_display.display( input_image );
        }
        else
        {
            grab_image( input_image );
            my_display.display( input_image );
        }

        my_display.wait();

    } while( !my_display.is_closed() );

    return 0;
}
