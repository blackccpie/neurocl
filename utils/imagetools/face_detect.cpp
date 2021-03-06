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

#include "face_detect.h"

extern "C" {
#include "ccv.h"
}

#include "CImg.h"

#include <iostream>

using namespace cimg_library;

class face_detect::face_detect_impl
{
public:

    face_detect_impl( int speedup_factor = 1 ) : m_speedup_factor( speedup_factor ), m_image( nullptr ), m_cascade( nullptr )
    {
        m_cascade = ccv_scd_classifier_cascade_read( "../../ccv/samples/face.sqlite3" );
    }

    virtual ~face_detect_impl()
    {
        ccv_scd_classifier_cascade_free( m_cascade );
    }

	template<typename T>
    const std::vector<face_detect::face_rect>& detect( const CImg<T>& image )
    {
        m_face_rects.clear();

        // TEMPORARY HACK!!!!!
        // FIND A WAY TO WORK WITH COLOR RGB IMAGE!
        CImg<unsigned char> grey_image = image.get_channel(0);

		if ( m_speedup_factor > 1 )
			grey_image.resize( grey_image.width()/m_speedup_factor, grey_image.height()/m_speedup_factor );

        ccv_read( grey_image.data(), &m_image, CCV_IO_GRAY_RAW | CCV_IO_NO_COPY, grey_image.height(), grey_image.width(), grey_image.width() );

        ccv_array_t* faces = ccv_scd_detect_objects( m_image, &m_cascade, 1, ccv_scd_default_params );

        std::cout << "Detected " << faces->rnum << " faces!" << std::endl;

        for ( int i = 0; i < faces->rnum; i++ )
        {
            ccv_comp_t* face = (ccv_comp_t*)ccv_array_get( faces, i );

            face_detect::face_rect face_rec( m_speedup_factor*face->rect.x, m_speedup_factor*face->rect.y,
                m_speedup_factor*(face->rect.x + face->rect.width), m_speedup_factor*(face->rect.y + face->rect.height) );

            m_face_rects.push_back( face_rec );

            std::cout << face_rec.x0 << " " << face_rec.y0 << " " << face_rec.x1 << " " << face_rec.y1
                << " (" << face->rect.width << "/" << face->rect.height << ")"
                << " face ratio " << ( (float)(face->rect.height) / (float)(face->rect.width) ) << std::endl;
        }

        ccv_array_free( faces );
        ccv_matrix_free( m_image );

        return m_face_rects;
    }

private:

	int m_speedup_factor;

    std::vector<face_detect::face_rect> m_face_rects;

    ccv_dense_matrix_t* m_image;
    ccv_scd_classifier_cascade_t* m_cascade;
};

face_detect::face_detect() : m_face_detect_impl( new face_detect_impl( 2 ) ) // speedup x2
{
}

face_detect::~face_detect() = default;

template<typename T>
const std::vector<face_detect::face_rect>&  face_detect::detect( const T* input, const int sizeX, const int sizeY )
{
	// Copy cast to float
    const CImg<float> _input( CImg<T>( input, sizeX, sizeY, 1, 1, true /*shared*/) );
    return m_face_detect_impl->detect<T>( _input );
}

template const std::vector<face_detect::face_rect>&  face_detect::detect<float>( const float* input, const int sizeX, const int sizeY );
template const std::vector<face_detect::face_rect>&  face_detect::detect<unsigned char>( const unsigned char* input, const int sizeX, const int sizeY );
