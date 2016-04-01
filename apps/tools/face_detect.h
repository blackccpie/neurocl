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

#ifndef FACE_DETECT_H
#define FACE_DETECT_H

#include "CImg.h"

#include <boost/shared_ptr.hpp>

#include <vector>

class face_detect_impl;

// Class to manage face detection
class face_detect
{
public:

    struct face_rect
    {
        face_rect( int _x0, int _y0, int _x1, int _y1 )
            : x0( _x0 ), y0( _y0 ), x1( _x1 ), y1( _y1 ) {}

        int x0;
        int y0;
        int x1;
        int y1;
    };

public:
    face_detect();
    virtual ~face_detect();

    const std::vector<face_rect>& detect( cimg_library::CImg<float>& image );

private:

    boost::shared_ptr<face_detect_impl> m_face_detect_impl;
};

#endif //FACE_DETECT_H
