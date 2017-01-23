/*
The MIT License

Copyright (c) 2015-2017 Albert Murienne

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

#ifndef TENSOR_LOSS_FUNCTIONS_H
#define TENSOR_LOSS_FUNCTIONS_H

#include "tensor_operations.h"

namespace neurocl { namespace convnet { namespace tensor_loss_functions {

class mse
{
public:

    static tensor f( tensor& y, tensor& t )
    {
        float factor = 0.5f / static_cast<float>( y.size() );
        return std::move( factor * ( y - t ) * ( y - t ) );
    }

    static tensor d_f( tensor& y, tensor& t )
    {
        float factor = 1.f / static_cast<float>( y.size() );
        return std::move( factor * ( y - t ) );
    }
};

// cross-entropy loss function for (multiple independent) binary classifications, aka binary logistic regression
// better use with sigmoid activation : removes the 1/(1-y) term, and only relies on (y-t) error
class cross_entropy
{
public:

    /*static tensor f( tensor& y, tensor& t )
    {
    }*/

    static tensor d_f( tensor& y, tensor& t )
    {
        return std::move( ( y - t ) / ( y * ( 1.f - y ) /*+ 1e-10f*/ ) );
    }
};

// cross-entropy loss function for multi-class classification, aka negative log likelihood
// better use with softmax activation
class cross_entropy_multiclass
{
public:

    /*static tensor f( tensor& y, tensor& t )
    {
    }*/

    static tensor d_f( tensor& y, tensor& t )
    {
        return std::move( -t / y );
    }
};

class cross_entropy_softmax // should only be used with softmax activation
{
public:

    /*static tensor f( tensor& y, tensor& t )
    {
    }*/

    static tensor d_f( tensor& y, tensor& t )
    {
		// For detailed explanation of the multiclass cross entropy with softmax simplified equation:
		// http://www.ics.uci.edu/~pjsadows/notes.pdf
		// http://peterroelants.github.io/posts/neural_network_implementation_intermezzo02
        return std::move( y - t );
    }
};

} /*namespace neurocl*/ } /*namespace convnet*/ } /*namespace tensor_loss_functions*/

#endif //TENSOR_LOSS_FUNCTIONS_H
