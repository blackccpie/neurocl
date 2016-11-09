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

#include "tensor.h"

#include "common/network_exception.h"
#include "common/network_utils.h"

namespace neurocl { namespace convnet {

const std::string dump_mat( const matrixF& mat /*, boost::optional<std::string> label = boost::none*/ )
{
    std::string separator;
    std::stringstream ss;
    //ss << ( label ? label.get() : "" ) << std::endl;
    for( matrixF::const_iterator1 it1 = mat.begin1(); it1 != mat.end1(); ++it1 )
    {
        for( matrixF::const_iterator2 it2 = it1.begin(); it2 !=it1.end(); ++it2 )
        {
            ss << separator << *it2;
            separator = " ";
        }
        separator = "";
        ss << std::endl;
    }
    return ss.str();
}

template<class T>
inline void random_normal_init( T& container, const float stddev = 1.f )
{
    utils::rand_gaussian_generator rgg( 0.f, stddev );

    for( auto& element : container.data() )
    {
        element = rgg();
    }
}

void tensor::_assert_same_size( const tensor& t )
{
    if ( ( m_width != t.w() ) ||
        ( m_height != t.h() ) ||
        ( m_depth1 != t.d1() ) ||
        ( m_depth2 != t.d2() ) )
        throw network_exception( "inconsistent tensor size" );
}

const std::string tensor::dump( const size_t d1, const size_t d2 ) const
{
    return dump_mat( m_tensor_array[d1][d2] );
}

void tensor::resize( const size_t width, const size_t height, const size_t depth1, const size_t depth2, boost::optional<size_t> opt_rand_nin )
{
    m_width = width;
    m_height = height;
    m_depth1 = depth1;
    m_depth2 = depth2;

    m_tensor_array.resize( boost::extents[m_depth1][m_depth2] );
    for( auto _matrices : m_tensor_array )
        for( auto& _matrix : _matrices )
        {
            _matrix = matrixF( m_width, m_height, 0.f );
            if ( opt_rand_nin )
            {
                //cf. http://neuralnetworksanddeeplearning.com/chap3.html#weight_initialization
                random_normal_init( _matrix, 1.f / std::sqrt( static_cast<float>( opt_rand_nin.get() ) ) );
            }
        }
}

void tensor::uniform_fill( const float& val )
{
    tensor_foreach() {
        m_tensor_array[d1][d2] = boost::numeric::ublas::scalar_matrix<float>( m_width, m_height, val );
    }
}

void tensor::uniform_fill_random( const float& stddev )
{
    utils::rand_gaussian_generator rgg( 0.f, stddev );

    tensor_foreach() {
        m_tensor_array[d1][d2] = boost::numeric::ublas::scalar_matrix<float>( m_width, m_height, rgg() );
    }
}

// TODO-CNN : write rvalue ref equivalent
tensor tensor::operator +=( const tensor& other )
{
    _assert_same_size( other );

    tensor_foreach() {
        m_tensor_array[d1][d2] += other.c_m(d1,d2);
    }

    return std::move(*this);
}

// TODO-CNN : write rvalue ref equivalent
tensor tensor::operator -=( const tensor& other )
{
    _assert_same_size( other );

    tensor_foreach() {
        m_tensor_array[d1][d2] -= other.c_m(d1,d2);
    }

    return std::move(*this);
}

// TODO-CNN : write rvalue ref equivalent
tensor tensor::operator *( const float val )
{
    tensor_foreach() {
        m_tensor_array[d1][d2] *= val;
    }

    return std::move(*this);
}

// TODO-CNN : write rvalue ref equivalent
tensor tensor::operator /( const float val )
{
    tensor_foreach() {
        m_tensor_array[d1][d2] /= val;
    }

    return std::move(*this);
}

// TODO-CNN : write rvalue ref equivalent
tensor tensor::operator +( const float val )
{
    tensor_foreach() {
        m_tensor_array[d1][d2] += boost::numeric::ublas::scalar_matrix<float>( m_width, m_height, val );
    }

    return std::move(*this);
}

bool tensor::operator ==( const tensor& other ) const
{
    tensor_foreach() {
        if ( !boost::numeric::ublas::detail::equals(
            m_tensor_array[d1][d2], other.m_tensor_array[d1][d2],
            std::numeric_limits<matrixF::value_type>::epsilon(), std::numeric_limits<matrixF::value_type>::min() ) )
            return false;
    }
    return true;
}

void tensor::fill(  const size_t d1,
                    const size_t d2,
                    const size_t data_size,
                    const float* data )
{
    if ( data_size != ( m_width * m_height ) )
        throw network_exception( "invalid fill size!" );

    std::copy( data, data + data_size, m_tensor_array[d1][d2].data().begin() );
}

void tensor::fill(  const size_t d1,
                    const size_t d2,
                    float* data )
{
    std::copy(  m_tensor_array[d1][d2].data().begin(),
                m_tensor_array[d1][d2].data().begin() + ( m_width * m_height ),
                data );
}

void tensor::grouped_fill( const size_t data_size, const float* data )
{
    size_t _offset = 0;

    tensor_foreach() {

        auto input_mat_iter = m_tensor_array[d1][d2].data().begin();
        const size_t _size = m_width * m_height;

        std::copy( data + _offset, data + _offset + _size, input_mat_iter );

        _offset += _size;
    }
}

void tensor::grouped_fill( float* data )
{
    size_t _offset = 0;

    tensor_foreach() {

        auto input_mat_iter = m_tensor_array[d1][d2].data().begin();
        const size_t _size = m_width * m_height;

        std::copy( input_mat_iter, input_mat_iter + _size, data + _offset );

        _offset += _size;
    }
}

} /*namespace neurocl*/ } /*namespace convnet*/
