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
#include "common/network_random.h"

namespace neurocl { namespace convnet {

const std::string dump_mat( const matrixF& mat /*, boost::optional<std::string> label = boost::none*/ )
{
    std::string separator;
    std::stringstream ss;
    //ss << ( label ? label.get() : "" ) << std::endl;
    for( matrixF::const_iterator1 it1 = mat.cbegin1(); it1 != mat.cend1(); ++it1 )
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
    random::rand_gaussian_generator rgg( 0.f, stddev );

    for( auto& element : container.data() )
    {
        element = rgg();
    }
}

void tensor::assert_same_size( const tensor& t )
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

tensor::tensor( const tensor&& t )
{
    m_width = t.m_width;
    m_height = t.m_height;
    m_depth1 = t.m_depth1;
    m_depth2 = t.m_depth2;

    m_tensor_array = std::move( t.m_tensor_array );
}

tensor::tensor( const tensor& t )
{
    m_width = t.m_width;
    m_height = t.m_height;
    m_depth1 = t.m_depth1;
    m_depth2 = t.m_depth2;

    m_tensor_array.resize( boost::extents[t.m_depth1][t.m_depth2] );
    m_tensor_array = t.m_tensor_array;
}

tensor& tensor::operator=( tensor&& other )
{
    m_width = other.m_width;
    m_height = other.m_height;
    m_depth1 = other.m_depth1;
    m_depth2 = other.m_depth2;

    m_tensor_array = std::move( other.m_tensor_array );

    return *this;
}

tensor& tensor::operator=( const tensor& other )
{
    m_width = other.m_width;
    m_height = other.m_height;
    m_depth1 = other.m_depth1;
    m_depth2 = other.m_depth2;

    m_tensor_array.resize( boost::extents[m_depth1][m_depth2] );
    m_tensor_array = other.m_tensor_array;

    return *this;
}

void tensor::resize( const size_t width, const size_t height, const size_t depth1, const size_t depth2 )
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
        }
}

void tensor::fill_random( const size_t& rand_nin )
{
    for( auto _matrices : m_tensor_array )
        for( auto& _matrix : _matrices )
        {
            /* http://cs231n.github.io/neural-networks-2/ cf. end Summary */
            _matrix = matrixF( m_width, m_height, 0.f );
            random_normal_init( _matrix, std::sqrt( 2.f / static_cast<float>( rand_nin ) ) );
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
    random::rand_gaussian_generator rgg( 0.f, stddev );

    tensor_foreach() {
        m_tensor_array[d1][d2] = boost::numeric::ublas::scalar_matrix<float>( m_width, m_height, rgg() );
    }
}

tensor tensor::operator +=( const tensor& other )
{
    assert_same_size( other );

    tensor_foreach() {
        m_tensor_array[d1][d2] += other.m_tensor_array[d1][d2];
    }

    // returning with std::move would inhibit RVO
    // thread about this:
    // http://stackoverflow.com/questions/4986673/c11-rvalues-and-move-semantics-confusion-return-statement
    return *this;
}

tensor tensor::operator -=( const tensor& other )
{
    assert_same_size( other );

    tensor_foreach() {
        m_tensor_array[d1][d2] -= other.m_tensor_array[d1][d2];
    }

    return *this;
}

tensor tensor::operator *( const float val )
{
    tensor_foreach() {
        m_tensor_array[d1][d2] *= val;
    }

    return *this;
}

tensor tensor::operator /( const float val )
{
    tensor_foreach() {
        m_tensor_array[d1][d2] /= val;
    }

    return *this;
}

tensor tensor::operator +( const float val )
{
    tensor_foreach() {
        m_tensor_array[d1][d2] += boost::numeric::ublas::scalar_matrix<float>( m_width, m_height, val );
    }

    return *this;
}

tensor tensor::operator -( const float val )
{
    tensor_foreach() {
        m_tensor_array[d1][d2] -= boost::numeric::ublas::scalar_matrix<float>( m_width, m_height, val );
    }

    return *this;
}

tensor tensor::operator -()
{
    tensor_foreach() {
        m_tensor_array[d1][d2] = -m_tensor_array[d1][d2];
    }

    return *this;
}

tensor tensor::operator -() const
{
    tensor output(*this);

    tensor_foreach() {
        output.m_tensor_array[d1][d2] = -m_tensor_array[d1][d2];
    }

    return output;
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

float tensor::norm1() const
{
    float _acc = 0.f;
    tensor_foreach() {
        std::for_each(m_tensor_array[d1][d2].data().begin(), m_tensor_array[d1][d2].data().end(),
            [&_acc] ( float a ) {
                _acc += std::abs( a );
            });
    }
    return _acc;
}

float tensor::norm2() const
{
    float _acc = 0.f;
    tensor_foreach() {
        std::for_each(m_tensor_array[d1][d2].data().begin(), m_tensor_array[d1][d2].data().end(),
            [&_acc] ( float a ) {
        		_acc += a * a;
            });
    }
    return std::sqrt( _acc );
}

float tensor::sum() const
{
    float _acc = 0.f;
    tensor_foreach() {
        std::for_each(m_tensor_array[d1][d2].data().begin(), m_tensor_array[d1][d2].data().end(),
            [&_acc] ( float a ) {
                _acc += a;
            });
    }
    return _acc;
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
