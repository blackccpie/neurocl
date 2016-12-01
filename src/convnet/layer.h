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

#ifndef LAYER_H
#define LAYER_H

#include "tensor_activations.h"

namespace neurocl { namespace convnet {

class tensor_solver_iface;

class layer
{

public:

     virtual const std::string type() const = 0;

    //! SIZING OF THE FEATURE MAPS
    size_t size() const { return width()*height()*depth(); }

    virtual size_t width() const = 0;
    virtual size_t height() const = 0;
    virtual size_t depth() const = 0;

    //! SIZING OF THE MODEL PARAMETERS
    virtual size_t nb_weights() const = 0;
    virtual size_t nb_bias() const = 0;

    virtual const tensor& feature_maps() const = 0;

    virtual void feed_forward() = 0;
    virtual void back_propagate() = 0;
    virtual void update_gradients() = 0;
    virtual void clear_gradients() = 0;
    virtual void gradient_descent( const std::shared_ptr<tensor_solver_iface>& solver ) = 0;

    //! Fill weights
    virtual void fill_w( const size_t data_size, const float* data ) = 0;
    virtual void fill_w( float* data ) = 0;

    //! Fill bias
    virtual void fill_b( const size_t data_size, const float* data ) = 0;
    virtual void fill_b( float* data ) = 0;

    //! Set training flag
    static void set_training( bool training ) { m_training = training; }

private:

    friend class pool_layer;
    friend class dropout_layer;
    template <class T> friend class conv_layer;
    template <class T> friend class full_layer;
    template <class T1,class T2> friend class output_layer;

    virtual tensor& error_maps() = 0;

protected:

    static bool m_training;
};

bool layer::m_training = false;

} /*namespace neurocl*/ } /*namespace convnet*/

#endif //LAYER_H
