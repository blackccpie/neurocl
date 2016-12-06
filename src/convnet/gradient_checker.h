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

#ifndef GRADIENT_CHECKER_H
#define GRADIENT_CHECKER_H

namespace neurocl { namespace convnet {

class tensor_solver_iface;

// TODO-CNN : move someday to a common templated class...
class gradient_checker
{
public:
    gradient_checker( tensor& weights, tensor& deltas ) : m_weights( weights ), m_deltas( deltas ) {}
    virtual ~gradient_checker() {}

    size_t size() {}

    void mod_plus() {}
    void mod_minus() {}
    void restore() {}
    void cost() {}
    void next() {}
    void error() {}
private:
    tensor& m_weights;
    tensor& m_deltas;
};

} /*namespace neurocl*/ } /*namespace convnet*/

#endif //GRADIENT_CHECKER_H
