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

#ifndef TENSOR_SOLVER_H
#define TENSOR_SOLVER_H

#include "solver.h"

namespace neurocl { namespace convnet {

class tensor_solver_iface
{
public:
    virtual void set_size( const size_t& size ) = 0;

    virtual void update( tensor& input, tensor& input_momentum, const tensor& gradient ) = 0;
    virtual void update_redux( tensor& input, tensor& input_momentum, const tensor& gradient ) = 0;
};

template<class solverT>
class tensor_solver : public tensor_solver_iface
{
public:
    tensor_solver( const float alpha, const float lambda, const float mu = 0 ) : m_solver( alpha, lambda, mu ) {}
    virtual ~tensor_solver() {}

    void set_size( const size_t& size ) override
    {
        m_solver.set_size( size );
    }

    void update( tensor& input, tensor& input_momentum, const tensor& gradient ) override
    {
        m_solver.update( input, input_momentum, gradient );
    }

    void update_redux( tensor& input, tensor& input_momentum, const tensor& gradient ) override
    {
        m_solver.update_redux( input, input_momentum, gradient );
    }

private:
    solverT m_solver;
};

} /*namespace neurocl*/ } /*namespace convnet*/

#endif //TENSOR_SOLVER_H
