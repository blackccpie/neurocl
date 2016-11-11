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

#include "tensor_operations.h"

#include "common/network_exception.h"

#include <cmath>

namespace neurocl { namespace convnet {

using nto = neurocl::convnet::tensor_operation;

/* Stochastic Gradient Descent solver implementation */
class solver_sgd
{
public:
    solver_sgd( const float alpha, const float lambda, const float mu )
        : m_set_size( 1 ), m_alpha( alpha ), m_lambda( lambda ), m_mu( mu ) {}
    virtual ~solver_sgd() {}

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

        input_momentum = ( m_mu * input_momentum ) - m_alpha * ( invm * gradient + m_lambda * input );
        input += input_momentum;
    }

    template<typename T>
    void update_redux( T& input, T& input_momentum, const T& gradient )
    {
        auto invm = 1.f / static_cast<float>( m_set_size );

        input_momentum = ( m_mu * input_momentum ) - m_alpha * ( invm * gradient );
        input += input_momentum;
    }

private:

    size_t m_set_size;

    float m_alpha;  // learning rate
    float m_lambda; // weight decay
    float m_mu;     //momentum
};

/* RMSprop solver implementation */
class solver_rms_prop
{
public:
    solver_rms_prop() : m_mu( 0.99f ), m_alpha( 0.0001f ), m_eps( 1e-8f ) {}
    virtual ~solver_rms_prop() {}

    template<typename T>
    void update( T& input, T& input_momentum, const T& gradient )
    {
        input_momentum = m_mu * input_momentum + ( 1 - m_mu ) * gradient * gradient;
        input -= m_alpha * gradient / std::sqrt( input_momentum + m_eps );
    }

private:

    float m_mu;         // decay term
    float m_alpha;      // learning rate
    const float m_eps;  // constant value to avoid zero-division
};

} /*namespace neurocl*/ } /*namespace convnet*/

#endif //SOLVER_H
