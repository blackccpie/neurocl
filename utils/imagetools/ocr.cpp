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

#include "ocr.h"

#include "neurocl.h"

#include "autothreshold.h"
#include "edge_detect.h"

#include "CImg.h"

#include <iostream>

using namespace neurocl;
using namespace cimg_library;

using t_digit_interval = std::pair<size_t,size_t>;

CImg<> get_row_sums( const CImg<>& input );
CImg<> get_line_sums( const CImg<>& input );
CImg<float> get_cropped_numbers( const CImg<float>& input );
void compute_ranges( const CImg<float>& input, std::vector<t_digit_interval>& number_intervals );
void center_number( CImg<float>& input );

class ocr_helper::ocr_helper_impl
{
public:
    ocr_helper_impl( std::shared_ptr<network_manager_interface> net_manager )
        : m_net_manager( net_manager ) {}
    virtual ~ocr_helper_impl() = default;

    template<typename T>
    void process( const T* input, const int sizeX, const int sizeY )
    {
        // Copy cast to float image
        const CImg<float> _input( CImg<T>( input, sizeX, sizeY, 1, 1, true /*shared*/) );

        m_cropped_numbers = get_cropped_numbers( _input );

        m_cropped_numbers.normalize( 0, 255 );

        auto_threshold( m_cropped_numbers.data(), m_cropped_numbers.width(), m_cropped_numbers.height() );

        //m_cropped_numbers.display();

        std::vector<t_digit_interval> number_intervals;
        compute_ranges( m_cropped_numbers, number_intervals );

        std::shared_ptr<samples_augmenter> smp_augmenter = std::make_shared<samples_augmenter>( 28, 28 );

        float output[10] = { 0.f };

        std::cout << "ocr_helper::process - started inferring numbers on detected intervals" << std::endl;

        for ( auto& ni : number_intervals )
        {
            if ( ( ni.second - ni.first ) < 10 ) // letter is thinner than 10px, too small!!
            {
                std::cout << "ocr_helper::process - digit is too thin, skipping..." << std::endl;
                continue;
            }

            std::cout << "ocr_helper::process - cropping at " << ni.first << " " << ni.second << std::endl;
            CImg<float> cropped_number( m_cropped_numbers.get_columns( ni.first, ni.second ) );

            std::cout << "ocr_helper::process - centering number" << std::endl;
            center_number( cropped_number );

            sample sample( cropped_number.width() * cropped_number.height(), cropped_number.data(), 10, output );
            m_net_manager->compute_augmented_output( sample, smp_augmenter );

            std::cout << "ocr_helper::process - max comp idx: " << sample.max_comp_idx() << " max comp val: " << sample.max_comp_val() << std::endl;

            m_recognitions.emplace_back( reco{ ni.first, sample.max_comp_idx(), 100.f * sample.max_comp_val() } );

            //cropped_number.display();
        }

        std::cout << "ocr_helper::process - ended inferring numbers on detected intervals" << std::endl;
    }

    template<typename imageT>
    const imageT& cropped_numbers()
    {
        return m_cropped_numbers;
    }

public:

    const std::vector<ocr_helper::reco>& recognitions() { return m_recognitions; }

    std::string reco_string()
    {
        std::string _str;
        for ( auto& _reco : m_recognitions )
            _str += std::to_string( _reco.value );
        return _str;
    }

private:

    std::vector<ocr_helper::reco> m_recognitions;
    CImg<float> m_cropped_numbers;
    std::shared_ptr<network_manager_interface> m_net_manager;
};

ocr_helper::ocr_helper( std::shared_ptr<network_manager_interface> net_manager )
    : m_pimpl( new ocr_helper_impl( net_manager ) )
{
}

ocr_helper::~ocr_helper() = default;

template<typename T>
void ocr_helper::process( const T* input, const int sizeX, const int sizeY )
{
    m_pimpl->process( input, sizeX, sizeY );
}

template void ocr_helper::process<unsigned char>( const unsigned char* input, const int sizeX, const int sizeY );
template void ocr_helper::process<float>( const float* input, const int sizeX, const int sizeY );

template<typename imageT>
const imageT& ocr_helper::cropped_numbers()
{
    return m_pimpl->cropped_numbers<imageT>();
}

template const CImg<float>& ocr_helper::cropped_numbers<CImg<float>>();

const std::vector<ocr_helper::reco>& ocr_helper::recognitions()
{
    return m_pimpl->recognitions();
}

std::string ocr_helper::reco_string()
{
    return m_pimpl->reco_string();
}

//********************** UTILITY FUNCTIONS **********************//

CImg<> get_row_sums( const CImg<>& input )
{
    // Compute row sums image
    CImg<float> row_sums( input.width(), 1 );
    cimg_forX( row_sums, x )
    {
        auto rsum = 0.f;
        cimg_for_insideY( input, y, 0 ) // 0px margin
        {
            rsum += input(x,y);
        }
        row_sums(x) = rsum;
    }
    return row_sums;
}

CImg<> get_line_sums( const CImg<>& input )
{
    // Compute line sums image
    CImg<float> line_sums( 1, input.height() );
    cimg_forY( line_sums, y )
    {
        auto lsum = 0.f;
        cimg_for_insideX( input, x, 0 ) // 0px margin
        {
            lsum += input(x,y);
        }
        line_sums(y) = lsum;
    }
    return line_sums;
}

CImg<float> get_cropped_numbers( const CImg<float>& input )
{
    CImg<unsigned char> work( input );
    CImg<unsigned char> work_edge( work );

    sobel_ccv::process<unsigned char>( work, work_edge );

    work_edge.normalize( 0, 255 );

    std::cout << "ocr_helper::get_cropped_numbers - image mean value is " << work_edge.mean() << " , noise variance is " << work_edge.variance_noise() << std::endl;

    if ( work_edge.variance_noise() > 10.f )
    {
    	work_edge.erode( 3 );
    	std::cout << "ocr_helper::get_cropped_numbers - post erosion mean value is " << work_edge.mean() << " , post erosion noise variance is " << work_edge.variance_noise() << std::endl;
    }

    work_edge.threshold( 40 );
    //work_edge.display();

    // Compute row sums image
    CImg<float> row_sums = get_row_sums( work_edge );
    row_sums.threshold( 5.f );
    //row_sums.display();

    // Compute line sums image
    CImg<float> line_sums = get_line_sums( work_edge );
    line_sums.threshold( 5.f );
    //line_sums.display();

    // Compute extraction coords
    int startX = 0;
    int stopX = 0;
    cimg_forX( row_sums, x )
    {
        if ( row_sums(x) )
        {
            startX = x;
            break;
        }
    }
    cimg_forX( row_sums, x )
    {
        if ( row_sums( row_sums.width() - x - 1 ) )
        {
            stopX = row_sums.width() - x - 1;
            break;
        }
    }

    int startY = 0;
    int stopY = 0;
    cimg_forY( line_sums, y )
    {
        if ( line_sums(y) )
        {
            startY = y;
            break;
        }
    }
    cimg_forY( line_sums, y )
    {
        if ( line_sums( line_sums.height() - y - 1 ) )
        {
            stopY = line_sums.height() - y - 1;
            break;
        }
    }

    int margin = ( stopY - startY ) / 7; // empirical ratio...
    startX -= 2 * margin;
    startY -= margin;
    stopX += 2 * margin;
    stopY += margin;

    // check boundaries
    startX = std::max( startX, 0 );
    startY = std::max( startY, 0 );
    stopX = std::min( stopX, input.width()-1 );
    stopY = std::min( stopY, input.height()-1 );

    std::cout << margin << " / " << startX << " " << startY << " " << stopX << " " << stopY << std::endl;

    CImg<float> cropped( input.get_crop( startX, startY, stopX, stopY ) );
    cropped = 1.f - cropped;

    return cropped;
}

void compute_ranges( const CImg<float>& input, std::vector<t_digit_interval>& number_intervals )
{
    // Compute row sums image
    CImg<float> row_sums = get_row_sums( input );

    row_sums.threshold( 1.f );

   //row_sums.display();

    // Detect letter ranges
    size_t first = 0;
    bool last_val = false;
    cimg_forX( row_sums, x )
    {
        bool cur_val = row_sums[x] > 0.f;

        if ( cur_val != last_val )
        {
            if ( !last_val ) // ascending front
            {
                first = x;
            }
            else //descending front
            {
                if ( first != 0 )
                {
                    std::cout << "compute_ranges - detected interval " << first << " " << x << std::endl;
                    number_intervals.push_back( std::make_pair( first, x ) );
                    first = 0;
                }
            }
        }
        last_val = cur_val;
    }
}

void center_number( CImg<float>& input )
{
    // Compute row sums image
    CImg<float> row_sums = get_row_sums( input );
    row_sums.threshold( 2.f );
    //row_sums.display();

    int startX = 0;
    int stopX = 0;
    bool last_val = false;
    cimg_forX( row_sums, x )
    {
        bool cur_val = row_sums[x] > 0.f;

        if ( cur_val != last_val )
        {
            if ( !last_val ) // ascending front
                startX = x;
            else //descending front
            {
                stopX = x;
                break;
            }
        }
        last_val = cur_val;
    }

    // Compute line sums image
    CImg<float> line_sums = get_line_sums( input );
    line_sums.threshold( 2.f );
    //line_sums.display();

    int startY = 0;
    int stopY = 0;
    last_val = false;
    cimg_forY( line_sums, y )
    {
        bool cur_val = line_sums[y] > 0.f;

        if ( cur_val != last_val )
        {
            if ( !last_val ) // ascending front
                startY = y;
            else //descending front
            {
                stopY = y;
                break;
            }
        }
        last_val = cur_val;
    }

    std::cout << "center_number - startX/stopX - " << startX << "/" << stopX << " startY/stopY - " << startY << "/" << stopY << std::endl;

    if ( ( stopX <= startX ) || ( stopY <= startY ) )
    {
        std::cout << "center_number - invalid centering request..." << std::endl;
        return;
    }

    // try to prepare image like MNIST does:
    // http://yann.lecun.com/exdb/mnist/

    //int max_dim = std::max( input.width(), input.height() );
    int max_dim = std::max( stopX - startX, stopY - startY );

    input.crop( startX, startY, stopX, stopY );
    input.resize( max_dim, max_dim, -100, -100, 0, 0, 0.5f, 0.5f );
    input.resize( 20, 20, -100, -100, 6 );
    input.normalize( 0, 255 );

    // compute center of mass
    int massX = 0;
    int massY = 0;
    int num = 0;
    cimg_forXY( input, x, y )
    {
        massX += input( x, y ) * x;
        massY += input( x, y ) * y;
        num += input( x, y );
    }
    massX /= num;
    massY /= num;

    std::cout << "center_number - Mass center X=" << massX << " Y=" << massY << std::endl;

    input.resize( 28, 28, -100, -100, 0, 0, 1.f - ((float)massX)/20.f, 1.f - ((float)massY)/20.f );

    input.normalize( 0.f, 1.f );

    //input.display();
}
