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

#ifndef TENSOR_GRADIENT_CHECKER_H
#define TENSOR_GRADIENT_CHECKER_H

namespace neurocl { namespace convnet {

// TODO-CNN : move someday to a common generic templated class...
class tensor_gradient_checker
{
public:
    tensor_gradient_checker( tensor& weights, tensor& deltas )
    	: m_index( 0 ), m_stored( 0.f ), m_weights( weights ), m_deltas( deltas )
    {
        m_weights._assert_same_size( m_deltas );

        m_line_size = m_weights.w();
        m_group_size = m_weights.d2();
        m_base_size = m_weights.w() * m_weights.h();
    }
    virtual ~tensor_gradient_checker() {}

    size_t size()
    {
        return m_base_size * m_weights.d1() * m_weights.d2();
    }

    void mod( const float epsilon )
    {
        std::swap( m_stored, _get_value() );
        _get_value() = m_stored + epsilon;
    }
    void restore()
    {
        std::swap( m_stored, _get_value() );
    }
    void set_grad( const float grad )
    {
        _get_value() = grad;
    }
    void next() { ++m_index; }
    void error() {}
private:
    float& _get_value()
    {
        size_t mod = m_index % m_base_size;
        size_t y = mod / m_line_size;
        size_t x = mod % m_line_size;
        size_t d1 = ( m_index / m_base_size ) % m_group_size;
        size_t d2 = ( m_index / m_base_size ) / m_group_size;

        return m_weights.m(d1,d2)(x,y);
    }
private:
    float m_stored;
    tensor& m_weights;
    tensor& m_deltas;

    size_t m_group_size;
    size_t m_line_size;
    size_t m_base_size;
    size_t m_index;
};

} /*namespace neurocl*/ } /*namespace convnet*/

#endif //TENSOR_GRADIENT_CHECKER_H
