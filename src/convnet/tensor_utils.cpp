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

#include "tensor_utils.h"
#include "tensor.h"

#include "CImg.h"

#include <boost/format.hpp>

namespace neurocl { namespace convnet {

namespace tensor_utils {

void visualizer::dump_features( const std::string& prefix, const tensor& t )
{
    tensor_foreach_p( t.d1(), t.d2() ) {
        cimg_library::CImg<float> tensor_img( t.c_m(d1,d2,{}).data().begin(), t.w(), t.h(), 1, 1, true /*shared*/ );
        tensor_img.save( boost::str( boost::format{"%1%_%2%_%3%.png"} % prefix % d1 % d2 ).c_str() );
    }
}

} //namespace utils

} /*namespace neurocl*/ } /*namespace convnet*/
