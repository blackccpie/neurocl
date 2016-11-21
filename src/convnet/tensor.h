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

#ifndef TENSOR_H
#define TENSOR_H

#include <boost/multi_array.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/optional.hpp>

#include <memory>

template<typename Type>
using matrixT = typename boost::numeric::ublas::matrix<Type>;

template <typename Type>
using matrix2T = typename boost::multi_array<matrixT<Type>,2>;

using matrixF = matrixT<float>;
using matrix2F = matrix2T<float>;

namespace neurocl { namespace convnet {

namespace tensor_activations {
    class sigmoid;
    class relu;
    class softmax;
}

#define tensor_foreach() for ( auto d1 = 0; d1 < m_depth1; d1++ ) \
                            for ( auto d2 = 0; d2 < m_depth2; d2++ )

#define tensor_foreach_p(n1,n2) for ( auto d1 = 0; d1 < n1; d1++ ) \
                            for ( auto d2 = 0; d2 < n2; d2++ )

class tensor
{
public:
    tensor() : m_width(0), m_height(0), m_depth1(0), m_depth2(0) {}
    virtual ~tensor() {}

    bool empty() const { return ( m_depth1 == 0 ) && ( m_depth2 == 0 ); }

    // move constructor
    tensor( const tensor&& t )
    {
		// TODO-CNN
		// NEEDS A REWORK PASS WITH ASSIGNMENT OPERATORS
        m_width = t.m_width;
        m_height = t.m_height;
        m_depth1 = t.m_depth1;
        m_depth2 = t.m_depth2;

        m_tensor_array.resize( boost::extents[m_depth1][m_depth2] );

        tensor_foreach() {
            m_tensor_array[d1][d2] = std::move( t.m_tensor_array[d1][d2] );
        }
    }

    // copy constructor
    tensor( const tensor& t )
    {
		// TODO-CNN
		// NEEDS A REWORK PASS WITH ASSIGNMENT OPERATORS
        m_width = t.m_width;
        m_height = t.m_height;
        m_depth1 = t.m_depth1;
        m_depth2 = t.m_depth2;

        m_tensor_array.resize( boost::extents[t.m_depth1][t.m_depth2] );

        tensor_foreach() {
            m_tensor_array[d1][d2] = t.m_tensor_array[d1][d2];
        }
    }

    // move assignment operator
    tensor& operator=( tensor&& other )
    {
        m_width = other.m_width;
        m_height = other.m_height;
        m_depth1 = other.m_depth1;
        m_depth2 = other.m_depth2;

        m_tensor_array = std::move( other.m_tensor_array );

        return *this;
    }

    // assignment operator
    tensor& operator=( tensor& other )
    {
        m_width = other.m_width;
        m_height = other.m_height;
        m_depth1 = other.m_depth1;
        m_depth2 = other.m_depth2;

        m_tensor_array.resize( boost::extents[m_depth1][m_depth2] );

        tensor_foreach_p( other.d1(), other.d2() ) {
            m_tensor_array[d1][d2] = other.m_tensor_array[d1][d2];
        }

        return *this;
    }

    void resize( const tensor& other )
    {
        resize( other.w(), other.h(), other.d1(), other.d2() );
    }

    // TODO-CNN : name of the function doesn't tell the matrix will be set to 0
    void resize( const size_t width, const size_t height, const size_t depth1, const size_t depth2, boost::optional<size_t> opt_rand_nin = boost::none );

    void flip()
    {
        tensor_foreach() {
            std::reverse( m_tensor_array[d1][d2].data().begin(), m_tensor_array[d1][d2].data().end() );
        }
    }

    void clear()
    {
        tensor_foreach() {
            m_tensor_array[d1][d2].clear();
        }
    }

    size_t w() const { return m_width; }
    size_t h() const { return m_height; }
    size_t d1() const { return m_depth1; }
    size_t d2() const { return m_depth2; }

    // operators overload
    tensor operator +=( const tensor& other );
    tensor operator -=( const tensor& other );
    tensor operator *( const float val );
    tensor operator /( const float val );
    tensor operator +( const float val );
    bool operator ==( const tensor& other ) const;

    void uniform_fill( const float& val );
    void uniform_fill_random( const float& stddev );

    void fill(  const size_t d1,
                const size_t d2,
                const size_t data_size,
                const float* data );

    void fill(  const size_t d1,
                const size_t d2,
                float* data );

    void grouped_fill( const size_t data_size, const float* data );
    void grouped_fill( float* data );

    const std::string dump( const size_t d1, const size_t d2 ) const;

protected:

    friend class tensor_operation;
    friend class tensor_activations::sigmoid;
    friend class tensor_activations::relu;
    friend class tensor_activations::softmax;

    matrixF& m( const size_t d1, const size_t d2 )  { return m_tensor_array[d1][d2]; }
    const matrixF& c_m( const size_t d1, const size_t d2 ) const { return m_tensor_array[d1][d2]; }

private:

    void _assert_same_size( const tensor& t );

private:

    size_t m_width;
    size_t m_height;
    size_t m_depth1; // --> replication level of feature maps
    size_t m_depth2; // --> number of feature maps

    matrix2F m_tensor_array;
};

} /*namespace neurocl*/ } /*namespace convnet*/

#endif //TENSOR_H
