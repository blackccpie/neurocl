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

#ifndef SOLVER_H
#define SOLVER_H

#include "common/network_exception.h"

namespace neurocl { namespace convnet {

/* Stochastic Gradient Descent solver implementation */
class solver
{
public:
    solver( const float learning_rate, const float weight_decay, const float momentum )
        : m_set_size( 1 ), m_learning_rate( learning_rate ), m_weight_decay( weight_decay ), m_momentum( momentum ) {}
    virtual ~solver() {}

    void set_size( const size_t& size )
    {
        if ( !size )
            throw network_exception( "cannot set solver size to zero" );

        m_set_size = size;
    }

    template<typename T>
    void update( T& input, T& input_momentum, const T& gradient )
    {
        auto invm = 1.f / static_cast<float>( m_set_size );

        input_momentum = ( m_momentum * input_momentum ) - m_learning_rate * ( invm * gradient + m_weight_decay * input );
        input += input_momentum;
    }

    template<typename T>
    void update_redux( T& input, T& input_momentum, const T& gradient )
    {
        auto invm = 1.f / static_cast<float>( m_set_size );

        input_momentum = ( m_momentum * input_momentum ) - m_learning_rate * ( invm * gradient );
        input += input_momentum;
    }

private:

    size_t m_set_size;

    float m_learning_rate;  // [0.0..1.0]
    float m_weight_decay;   // [0.0..1.0]
    float m_momentum;       // [0.0..1.0]
};

} /*namespace neurocl*/ } /*namespace convnet*/

#endif //SOLVER_H
