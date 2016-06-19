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

#ifndef POOL_LAYER_BNU_H
#define POOL_LAYER_BNU_H

#include "layer_bnu.h"

namespace neurocl {

class pool_layer_bnu  : public layer_bnu
{
public:

    pool_layer_bnu();
	virtual ~pool_layer_bnu() {}

    void populate(  const layer_bnu* prev_layer,
                    const size_t width,
                    const size_t height,
                    const size_t depth );

    virtual bool has_feature_maps() const { return true; }

    virtual size_t width() const { return m_feature_maps[0].size1(); };
    virtual size_t height() const { return m_feature_maps[0].size2(); };
    virtual size_t depth() const { return m_feature_maps.shape()[0]; }

    virtual const vectorF& activations() const
        { return empty::vector; }
    virtual const matrixF& feature_map( const int depth ) const
        { return m_feature_maps[depth]; }

    virtual void feed_forward();
    virtual void back_propagate();

private:

    size_t m_subsample;

    const layer_bnu* m_prev_layer;

    marray1F m_feature_maps;
    marray1F m_error_maps;
};

} //namespace neurocl

#endif //POOL_LAYER_BNU_H
