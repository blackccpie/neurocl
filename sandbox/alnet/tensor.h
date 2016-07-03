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

class tensor
{
public:
    tensor() : m_width(0), m_height(0), m_depth1(0), m_depth2(0) {}
    virtual ~tensor() {}

    // move constructor
    tensor( const tensor&& t )
    {
        for ( auto d1 = 0; d1 < m_depth1; d1++ )
            for ( auto d2 = 0; d2 < m_depth2; d2++ )
                m_tensor_array[d1][d2] = t.m_tensor_array[d1][d2];
    }

    // copy constructor
    tensor( const tensor& t )
    {
        for ( auto d1 = 0; d1 < m_depth1; d1++ )
            for ( auto d2 = 0; d2 < m_depth2; d2++ )
                m_tensor_array[d1][d2] = t.m_tensor_array[d1][d2];
    }

    // move assignment operator
    tensor& operator=( tensor&& other )
    {
        *this = std::move( other );

        return *this;
    }

    // assignment operator
    tensor& operator=( tensor& other )
    {
        for ( auto d1 = 0; d1 < other.d1(); d1++ )
            for ( auto d2 = 0; d2 < other.d2(); d2++ )
                m_tensor_array[d1][d2] = other.m_tensor_array[d1][d2];

        return *this;
    }

    void resize( const size_t width, const size_t height, const size_t depth1, const size_t depth2 )
    {
        m_width = width;
        m_depth1 = depth1;
        m_depth2 = depth2;

        m_tensor_array.resize( boost::extents[m_depth1][m_depth2] );
        for( auto _matrices : m_tensor_array )
            for( auto _matrix : _matrices )
                _matrix = matrixF( m_width, m_height );
    }

    size_t w() const { return m_width; }
    size_t h() const { return m_height; }
    size_t d1() const { return m_depth1; }
    size_t d2() const { return m_depth2; }

protected:

    friend class tensor_operation;

    matrixF& m( size_t i, size_t j )  { return m_tensor_array[i][j]; }
    const matrixF& c_m( size_t i, size_t j ) const { return m_tensor_array[i][j]; }

private:

    size_t m_width;
    size_t m_height;
    size_t m_depth1;
    size_t m_depth2;

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

public:

    static tensor muladd( const tensor& inputA, const tensor& inputB, const tensor& inputC );

    static void relu( tensor& input );

    static void d_relu( tensor& input, const tensor& output );

    template<kernel_mode km, pad_mode pm>
    static tensor convolve_add( const tensor& input, const tensor& filter, const int stride );

    static void optimize( std::shared_ptr<optimizer> optimizer, tensor& input, tensor& deltas )
    {

    }
};

} //namespace neurocl

#endif //TENSOR_H
