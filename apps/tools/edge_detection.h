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

#include "CImg.h"

#include <iostream>
#include <type_traits>

#include <boost/multi_array.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/math/special_functions/pow.hpp>

using namespace cimg_library;

// use with normalized [0,1] floating point images
template<typename T>
void sobel( const CImg<T>& image_in, CImg<T>& image_out )
{
	static_assert( boost::is_same<T,float>::value || boost::is_same<T,double>::value,
		"Template type should be floating point type!" );

	T upper_bound = 1;
	T lower_bound = 0;
	T sum;
	T sumX, sumY;

	T GX[3][3];
	T GY[3][3];

	//Sobel Matrices Horizontal
	GX[0][0] = 1; GX[0][1] = 0; GX[0][2] = -1;
	GX[1][0] = 2; GX[1][1] = 0; GX[1][2] = -2;
	GX[2][0] = 1; GX[2][1] = 0; GX[2][2] = -1;

	//Sobel Matrices Vertical
	GY[0][0] =  1; GY[0][1] =	 2; GY[0][2] =   1;
	GY[1][0] =  0; GY[1][1] =	 0; GY[1][2] =   0;
	GY[2][0] = -1; GY[2][1] =	-2;	GY[2][2] =  -1;

	/*Edge detection using Sobel Algorithm*/

	for( int y = 0; y < image_in.height() ; y++)
	{
		for( int x = 0; x < image_in.width() ; x++)
		{
			sumX	= 0;
			sumY	= 0;

			/*Image Boundaries*/
			if( y == 0 || y == image_in.height() - 1 )
				sum = 0;
			else if( x == 0 || x == image_in.width() - 1 )
				sum = 0;
			else
			{
				/*Convolution for X*/
				for( int i = -1; i < 2; i++ )
				{
					for( int j = -1; j < 2; j++ )
					{
						sumX = sumX + GX[j+1][i+1] * image_in(x+j,y+i);
					}
				}

				/*Convolution for Y*/
				for( int i = -1; i < 2; i++ )
				{
					for( int j = -1; j < 2; j++ )
					{
						sumY = sumY + GY[j+1][i+1] * image_in(x+j,y+i);
					}
				}

				/*Edge strength*/
				sum = std::sqrt( boost::math::pow<2>( sumX ) + boost::math::pow<2>( sumY ) );
			}

			if(sum > upper_bound) sum = upper_bound;
			if(sum < lower_bound) sum = lower_bound;

			image_out(x,y) = sum;//( upper_bound - sum );

			//std::cout << "x " << x << " y " << y << " SUM " << image_out(x,y) << std::endl;
		}
	}
}

template<typename T>
class canny
{
public:
	canny( const int& rows, const int& columns );
	virtual ~canny();

	void process( const cimg_library::CImg<T>& image_in, cimg_library::CImg<T>& image_out );

private:

	void _gaussian_blur( const cimg_library::CImg<T>& image_in, cimg_library::CImg<T>& image_out );
    void _sobel( cimg_library::CImg<T>& image );
    void _no_max( cimg_library::CImg<T>& image );
    void _hysteresis( cimg_library::CImg<T>& image );

private:

	T m_low_thresh;
	T m_high_thresh;

	unsigned int m_rows;
	unsigned int m_columns;

	boost::multi_array<int,2>  m_thetas;
	boost::multi_array<T,2> m_mag_array;
};
