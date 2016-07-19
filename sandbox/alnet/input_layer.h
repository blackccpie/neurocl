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

#ifndef INPUT_LAYER_H
#define INPUT_LAYER_H

#include "layer.h"

namespace neurocl {

class input_layer : public layer
{
public:

    input_layer() {}
	virtual ~input_layer() {}

    void populate(  const size_t width,
                    const size_t height,
                    const size_t depth )
    {
        std::cout << "populating input layer " << std::endl;

        m_feature_maps.resize( width, height, 1, depth );
        m_error_maps.resize( width, height, 1, depth );
    }

    virtual size_t width() const override { return m_feature_maps.w(); };
    virtual size_t height() const override { return m_feature_maps.h(); };
    virtual size_t depth() const override { return m_feature_maps.d2(); }

    virtual const tensor& feature_maps() const override
        { return m_feature_maps; }

    virtual void prepare_training() override { /*NOTHING TO DO YET*/ }
    virtual void feed_forward() override { /*NOTHING TO DO YET*/ }
    virtual void back_propagate() override { /*NOTHING TO DO YET*/ }
    virtual void gradient_descent( const std::shared_ptr<optimizer>& optimizer ) override { /*NOTHING TO DO YET*/ }

protected:

    virtual tensor& error_maps() override
        { return m_error_maps; }

private:

    tensor m_feature_maps;
    tensor m_error_maps;
};

} //namespace neurocl

#endif //INPUT_LAYER_BNU_H
