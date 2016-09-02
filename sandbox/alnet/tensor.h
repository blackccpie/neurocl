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

#include <memory>

typedef typename boost::numeric::ublas::matrix<float> matrixF;
typedef typename boost::multi_array<matrixF,2> matrix2F;

namespace neurocl {

class optimizer;

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
    void resize( const size_t width, const size_t height, const size_t depth1, const size_t depth2, bool rand = false );

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
    tensor operator /( const float val );
    bool operator ==( const tensor& other ) const
    {
        tensor_foreach() {
            if ( !boost::numeric::ublas::detail::equals(
                m_tensor_array[d1][d2], other.m_tensor_array[d1][d2],
                std::numeric_limits<matrixF::value_type>::epsilon(), std::numeric_limits<matrixF::value_type>::min() ) )
                return false;
        }
        return true;
    }

    void fill( const float& val )
    {
        tensor_foreach() {
            m_tensor_array[d1][d2] = boost::numeric::ublas::scalar_matrix<float>( m_width, m_height, val );
        }
    }

    void fill(  const size_t d1,
                const size_t d2,
                const size_t data_size,
                const float* data )
    {
        // TODO-CNN : size assert!

        std::copy( data, data + data_size, m_tensor_array[d1][d2].data().begin() );
    }

    void fill(  const size_t d1,
                const size_t d2,
                float* data )
    {
        std::copy(  m_tensor_array[d1][d2].data().begin(),
                    m_tensor_array[d1][d2].data().begin() + ( m_width * m_height ),
                    data );
    }

    const std::string dump( const size_t d1, const size_t d2 ) const;

protected:

    friend class tensor_operation;

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

class tensor_operation
{
public:

    enum kernel_mode
    {
        kernel_std = 0,
        kernel_flip
    };

    enum pad_mode
    {
        pad_valid = 0,
        pad_same,
        pad_full
    };

    enum optimize_mode
    {
        optim_std = 0,
        optim_redux
    };

public:

    // returns aB (scalar product)
    static tensor scale( const float& val, const tensor& input );

    // returns A + B
    static tensor add( const tensor& inputA, const tensor& inputB );

    // returns A - B
    static tensor sub( const tensor& inputA, const tensor& inputB );

    // groups all submatrixes into a single one
    static tensor group( const tensor& input );

    // ungroups input matrix into multiple ones
    static void ungroup( const tensor& input, tensor& output );

    // returns A.B (standard product)
    static tensor elemul( const tensor& inputA, const tensor& inputB );

    // returns A.B (element product)
    static tensor mul( const tensor& inputA, const tensor& inputB );

    // returns A.B + C
    static tensor muladd( const tensor& inputA, const tensor& inputB, const tensor& inputC );

    // returns trans(A).B
    static tensor multrans1( const tensor& inputA, const tensor& inputB );

    // returns A.trans(B)
    static tensor multrans2( const tensor& inputA, const tensor& inputB );

    static void sig( tensor& input );

    static tensor d_sig( const tensor& input );

    template<kernel_mode km, pad_mode pm>
    static tensor convolve( const tensor& input, const tensor& filter, const int stride );

    template<kernel_mode km, pad_mode pm>
    static tensor convolve_add( const tensor& input, const tensor& filter, const int stride );

    static tensor subsample( const tensor& input, const size_t subsample );

    static tensor d_subsample( const tensor& input, const tensor& input_ref, const size_t subsample );

    template<optimize_mode om>
    static void optimize( const std::shared_ptr<optimizer>& optimizer, tensor& input, const tensor& deltas );
};

inline tensor operator*( const float& val, const tensor& t )
{
    return std::move( tensor_operation::scale( val, t ) );
};

inline tensor operator+( const tensor& t1, const tensor& t2 )
{
    return std::move( tensor_operation::add( t1, t2 ) );
};

inline tensor operator-( const tensor& t1, const tensor& t2 )
{
    return std::move( tensor_operation::sub( t1, t2 ) );
};

} //namespace neurocl

#endif //TENSOR_H
