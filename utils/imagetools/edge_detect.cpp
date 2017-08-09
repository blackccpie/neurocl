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

#include "edge_detect.h"

#include <boost/type_traits/is_same.hpp>
#include <boost/math/special_functions/pow.hpp>

#include <iostream>

using namespace cimg_library;

extern "C" {
	#include "ccv.h"
}

////////////////////////////////////// SOBEL CCV /////////////////////////////////////////////////

template<> void sobel_ccv::process<unsigned char>( const CImg<unsigned char>& image_in, CImg<unsigned char>& image_out )
{
	ccv_dense_matrix_t* ccv_image_in =
		ccv_dense_matrix_new( image_in.height(), image_in.width(), CCV_8U | CCV_C1 | CCV_NO_DATA_ALLOC, (void*)image_in.data(), 0);
	ccv_image_in->step = image_in.width() * sizeof(unsigned char);

	ccv_dense_matrix_t* ccv_image_out =
		ccv_dense_matrix_new( image_out.height(), image_out.width(), CCV_8U | CCV_C1 | CCV_NO_DATA_ALLOC, (void*)image_out.data(), 0);
	ccv_image_out->step = image_out.width() * sizeof(unsigned char);

	ccv_sobel( ccv_image_in, &ccv_image_out, CCV_8U, 0, 1 );

	//CImg<float> test( ccv_image_out->data.f32, 50, 50, 1, 1 );
	//test.display();
}

////////////////////////////////////// CANNY CCV /////////////////////////////////////////////////

template<> void canny_ccv::process<unsigned char>( const CImg<unsigned char>& image_in, CImg<unsigned char>& image_out )
{
	// http://www.kerrywong.com/2009/05/07/canny-edge-detection-auto-thresholding/
	unsigned char mean = image_in.mean();
	unsigned char low_thresh = 0.66 * mean;
	unsigned char high_thresh = 1.33 * mean;

	ccv_dense_matrix_t* ccv_image_in =
		ccv_dense_matrix_new( image_in.height(), image_in.width(), CCV_8U | CCV_C1 | CCV_NO_DATA_ALLOC, (void*)image_in.data(), 0);
	ccv_image_in->step = image_in.width() * sizeof(unsigned char);

	ccv_dense_matrix_t* ccv_image_out =
		ccv_dense_matrix_new( image_out.height(), image_out.width(), CCV_8U | CCV_C1 | CCV_NO_DATA_ALLOC, (void*)image_out.data(), 0);
	ccv_image_out->step = image_out.width() * sizeof(unsigned char);

	ccv_canny( ccv_image_in, &ccv_image_out, CCV_8U, 1, low_thresh, high_thresh );
}

////////////////////////////////////// SOBEL /////////////////////////////////////////////////

// use with normalized [0,1] floating point images
template<typename T>
void sobel::process( const CImg<T>& image_in, CImg<T>& image_out )
{
	//static_assert( boost::is_same<T,float>::value || boost::is_same<T,double>::value,
	//	"Template type should be floating point type!" );

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

template void sobel::process<float>( const CImg<float>& image_in, CImg<float>& image_out );

////////////////////////////////////// CANNY /////////////////////////////////////////////////

#define MAX_SIZE 5

//***************************
// helper function that returns true if a>b and c
//***************************
bool is_first_max( int a, int b, int c )
{
	return ( a>b && a>c );
}

//***************************
// convolve is a general helper funciton that applies a convolution
// to the image and then returns the weighted sum so that
// it can replace whatever pixel we were just analyzing
//**************************
template<typename T,const int dim>
T convolve( const CImg<T>& image_in, T con[][MAX_SIZE], T divisor, int i, int j )
{
    int midx = dim/2;
    int midy = dim/2;

	T weighted_sum = 0;
	for( int x = i-midx; x < i + dim-midx; x++ )
	{
		for( int y = j-midy; y < j + dim-midy; y++ )
		{
			weighted_sum += divisor * con[x-i+midx][y-j+midy] * image_in(x,y);
		}
	}
	return weighted_sum;
}

//*****************************
//helper function that says whether arg is between a-b or c-d
//*****************************
bool is_between( float arg, float a, float b, float c, float d )
{
	return ( ( arg >= a && arg <= b ) || ( arg >= c && arg <= d ) );
}

//****************************
// buckets the thetas into 0, 45, 90, 135
//****************************
int get_orientation( float angle )
{
	if( is_between( angle, -22.5, 22.5, -180, -157.5 ) || is_between( angle, 157.5, 180, -22.5, 0 ) )
		return 0;
	if( is_between( angle, 22.5, 67.5, -157.5, -112.5 ) )
		return 45;
	if( is_between( angle, 67.5, 112.5, -112.5, -67.5) )
		return 90;
	if( is_between( angle, 112.5, 157.5, -67.5, -22.5  ) )
		return 135;

	return -1;
}

template<typename T>
canny<T>::canny( const int& columns, const int& rows )
	: m_rows( rows ), m_columns( columns ),
	m_low_thresh( 0 ), m_high_thresh( 0 ),
	m_thetas( boost::extents[columns][rows] ), m_mag_array( boost::extents[columns][rows] )
{
	static_assert( boost::is_same<T,float>::value || boost::is_same<T,double>::value,
		"Template type should be floating point type!" );
}

template canny<float>::canny( const int& rows, const int& columns );

template<typename T>
canny<T>::~canny()
{
}

template canny<float>::~canny();

//*****************************
// gaussian blur
// applies a gaussian blur via a convolution of a gaussian
// matrix with sigma = 1.4. hard-coded in.
// future development could generate the gauss matrix on the fly
//*****************************
template<typename T>
void canny<T>::_gaussian_blur( const CImg<T>& image_in, CImg<T>& image_out )
{
	// define gauss matrix
	T gauss_array[5][5] = {	{2, 4, 5, 4, 2},
							{4, 9, 12,9, 4},
							{5, 12, 15, 12, 5},
							{4, 9, 12,9, 4},
							{2, 4, 5, 4, 2} };

	T gauss_divisor = 1.0/159.0;
	T sum = 0.0;

	for( auto j=2U; j < m_rows-2; j++ )
	{
		for( auto i=2U; i < m_columns-2; i++ )
		{
			sum = convolve<T,5>( image_in, gauss_array, gauss_divisor, i, j );
			image_out(i,j) = sum;
		}
	}
}

//****************************
// Applies a sobel filter to find the gradient direction
// and magnitude. those values are then stored in thetas and magArray
// so that info can be used later for further analysis
//****************************
template<typename T>
void canny<T>::_sobel( CImg<T>& image )
{
	T G_x, G_y, G;
	T sobel_y[5][5] = {	{-1, 0, 1,0,0},
						{-2, 0, 2,0,0},
						{-1, 0, 1,0,0},
						{0, 0, 0, 0, 0},
						{0, 0, 0, 0, 0} };

	T sobel_x[5][5] = {	{1, 2, 1, 0, 0},
						{0, 0, 0, 0, 0},
						{-1, -2, -1, 0, 0},
						{0, 0, 0, 0, 0},
						{0, 0, 0, 0, 0} };

	for ( auto j = 1U; j < m_rows-1; j++ )
	{
		for ( auto i = 1U; i < m_columns-1; i++ )
		{
			G_x = convolve<T,3>( image, sobel_x, 1, i, j );
			G_y = convolve<T,3>( image, sobel_y, 1, i, j );
			G = std::sqrt( G_x*G_x + G_y*G_y );

			m_thetas[i][j] = get_orientation( 180.0 * std::atan2( G_y, G_x ) / cimg::PI );

			m_mag_array[i][j] = G;
		}
	}
}

//*****************************
//non-maximum suppression
//depending on the orientation, pixels are either thrown away or accepted
//by checking it's neighbors
//*****************************
template<typename T>
void canny<T>::_no_max( CImg<T>& image )
{
	for( auto j=1U ; j < m_rows-1 ; j++ )
	{
	    for( auto i=1U ; i < m_columns-1 ; i++ )
		{
			 	//std::cout << m_thetas[i][j] << std::endl;

				switch( m_thetas[i][j] )
				{
				case 0:
					if( is_first_max( m_mag_array[i][j], m_mag_array[i+1][j], m_mag_array[i-1][j] ) )
					{
						image(i,j) = 1; // white
					}
					else
					{
						image(i,j) = 0; // black
					}
				break;

				case 45:
					if( is_first_max( m_mag_array[i][j], m_mag_array[i+1][j+1], m_mag_array[i-1][j-1] ) )
					{
						image(i,j) = 1; // white
					}
					else
					{
						image(i,j) = 0; // black
					}
				break;

				case 90:
					if( is_first_max( m_mag_array[i][j], m_mag_array[i][j+1], m_mag_array[i][j-1] ) )
					{
						image(i,j) = 1; // white
					}
					else
					{
						image(i,j) = 0; // black
					}
				break;

				case 135:
					if( is_first_max( m_mag_array[i][j], m_mag_array[i+1][j-1], m_mag_array[i-1][j+1] ) )
					{
						image(i,j) = 1; // white
					}
					else
					{
						image(i,j) = 0; // black
					}
				break;

				default:
				break;
			}
		}
	}
}

//*******************************
//hysteresis noise filter makes lines continuous and filters out the noise
// see the pdf that we used to understand this step in english (Step 5)
//*******************************
template<typename T>
void canny<T>::_hysteresis( CImg<T>& image )
{
	bool greater_found;
	bool between_found;

	for( auto j=2U ; j < m_rows-2 ; j++ )
	{
		for( auto i=2U ; i < m_columns-2 ; i++ )
		{
			if( m_mag_array[i][j] < m_low_thresh )
			{
				image(i,j) = 0; // black
			}

			if( m_mag_array[i][j] > m_high_thresh )
			{
				image(i,j) = 1; // white
			}

			/*If pixel (x, y) has gradient magnitude between tlow and thigh and
			any of its neighbors in a 3 × 3 region around
			it have gradient magnitudes greater than thigh, keep the edge*/

			if( m_mag_array[i][j] >= m_low_thresh && m_mag_array[i][j] <= m_high_thresh)
			{
				greater_found = false;
				between_found = false;
				for( int m = -1; m < 2; m++ )
				{
					for( int n = -1; n < 2; n++ )
					{
						if( m_mag_array[i+m][j+n] > m_high_thresh )
						{
							image(i,j) = 0;
							greater_found = true;
				 		}
				 		if( m_mag_array[i][j] > m_low_thresh && m_mag_array[i][j] < m_high_thresh )
							between_found = true;
					}
				}

				if( !greater_found && between_found )
				{
					for( int m = -2; m < 3; m++ )
					{
						for( int n = -2; n < 3; n++ )
						{
							if( m_mag_array[i+m][j+n] > m_high_thresh )
								greater_found = true;
						}
					}
				}

				if( greater_found )
					image(i,j) = 0;
				else
					image(i,j) = 1;
			}
		}
	}
}

/*If pixel (x, y) has gradient magnitude between tlow and thigh and any of its neighbors in a 3 × 3 region around
it have gradient magnitudes greater than thigh, keep the edge (write out white).
If none of pixel (x, y)s neighbors have high gradient magnitudes but at least one falls between tlow and thigh,
search the 5 × 5 region to see if any of these pixels have a magnitude greater than thigh. If so, keep the edge
(write out white).
*/

template<typename T>
void canny<T>::process( const CImg<T>& image_in, CImg<T>& image_out )
{
	// http://www.kerrywong.com/2009/05/07/canny-edge-detection-auto-thresholding/
	T mean = image_in.mean();
	m_low_thresh = 0.66 * mean;
	m_high_thresh = 1.33 * mean;

	_gaussian_blur( image_in, image_out );
	_sobel( image_out );
	_no_max( image_out );
	_hysteresis( image_out );

	// 2px border not managed for now
	cimg_for_borderXY( image_out, x, y, 3 ) { image_out( x, y ) = 0; }
}

template void canny<float>::process( const CImg<float>& image_in, CImg<float>& image_out );
