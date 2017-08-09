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

#ifndef EDGE_DETECT_H
#define EDGE_DETECT_H

#include "CImg.h"

#include <boost/multi_array.hpp>

using namespace cimg_library;

class sobel
{
public:

	// use with normalized [0,1] floating point images
	template<typename T>
	static void process( const CImg<T>& image_in, CImg<T>& image_out );
};

template<typename T>
class canny
{
public:
	canny( const int& columns, const int& rows ); // width , height
	virtual ~canny();

	// use with normalized [0,1] floating point images
	void process( const cimg_library::CImg<T>& image_in, cimg_library::CImg<T>& image_out );

private:

	void _gaussian_blur( const cimg_library::CImg<T>& image_in, cimg_library::CImg<T>& image_out );
    void _sobel( cimg_library::CImg<T>& image );
    void _no_max( cimg_library::CImg<T>& image );
    void _hysteresis( cimg_library::CImg<T>& image );

private:

	unsigned int m_rows;
	unsigned int m_columns;

	T m_low_thresh;
	T m_high_thresh;

	boost::multi_array<int,2>  m_thetas;
	boost::multi_array<T,2> m_mag_array;
};

/*** CCV IMPLEMENTATION WRAPPERS ***/

class sobel_ccv
{
public:

	template<typename T>
	static void process( const CImg<T>& image_in, CImg<T>& image_out );
};

class canny_ccv
{
public:

	template<typename T>
	static void process( const CImg<T>& image_in, CImg<T>& image_out );
};

#endif //EDGE_DETECT_H
