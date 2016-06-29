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

#ifndef OPTIMIZER_H
#define OPTIMIZER_H

namespace neurocl {

class optimizer
{
public:
    optimizer( const float learning_rate, const float weight_decay )
        : m_learning_rate( learning_rate ), m_weight_decay( weight_decay ), m_set_size( 1 ) {}
    virtual ~optimizer() {}

    void set_size( const size_t& size )
    {
        m_set_size = size;
    }

    template<typename T>
    void update( T& input, const T& gradient )
    {
        auto invm = 1.f / static_cast<float>( m_set_size );

        input -= m_learning_rate * ( invm * gradient + m_weight_decay * input );
    }

    template<typename T>
    void update_redux( T& input, const T& gradient )
    {
        auto invm = 1.f / static_cast<float>( m_set_size );

        input -= m_learning_rate * ( invm * gradient );
    }

private:

    size_t m_set_size;

    float m_learning_rate; // [0.0..1.0]
    float m_weight_decay; // [0.0..1.0]
};

} //namespace neurocl

#endif //OPTIMIZER_H
