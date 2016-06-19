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

#ifndef LAYER_BNU_H
#define LAYER_BNU_H

#include "network_interface.h"

#include <boost/multi_array.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>

typedef typename boost::numeric::ublas::vector<float> vectorF;
typedef typename boost::numeric::ublas::matrix<float> matrixF;

typedef typename boost::multi_array<matrixF,1> marray1F;
typedef typename boost::multi_array<matrixF,2> marray2F;

namespace neurocl {

class layer_bnu
{

public:

    virtual bool is_input() { return false; }

    virtual bool has_feature_maps() const = 0;

    size_t size() const { return width()*height()*depth(); }
    virtual size_t width() const = 0;
    virtual size_t height() const = 0;
    virtual size_t depth() const = 0;

    virtual const vectorF& activations() const = 0;
    virtual const matrixF& feature_map( const int depth ) const = 0;

    virtual matrixF& error_map( const int depth ) const { empty::matrix; } // TODO-CN : temporary empty impl

    virtual void feed_forward() = 0;
    virtual void back_propagate() = 0;
    virtual void gradient_descent() {}; // TODO-CN : temporary empty impl

protected:

    struct empty
    {
        static const matrixF matrix;
        static const vectorF vector;
    };
};

} //namespace neurocl

#endif //LAYER_BNU_H
