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

#ifndef TENSOR_LOSS_FUNCTIONS_H
#define TENSOR_LOSS_FUNCTIONS_H

#include "tensor_operations.h"

namespace neurocl { namespace convnet { namespace tensor_loss_functions {

class mse
{
public:

    /*static tensor f( tensor& a, tensor& b )
    {
    }*/

    static tensor d_f( tensor& y, tensor& t )
    {
        return std::move( y - t );
    }
};

class cross_entropy_multiclass
{
public:

    /*static tensor f( tensor& a, tensor& b )
    {
    }*/

    static tensor d_f( tensor& y, tensor& t )
    {
        // TODO : rewrite as y * ( 1 - y )
        return std::move( ( y - t ) / ( y - y * y ) );
    }
};

} /*namespace neurocl*/ } /*namespace convnet*/ } /*namespace tensor_loss_functions*/

#endif //TENSOR_LOSS_FUNCTIONS_H
