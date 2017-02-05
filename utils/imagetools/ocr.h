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

#ifndef OCR_H
#define OCR_H

#include "neurocl.h"

#include "autothreshold.h"
#include "edge_detect.h"

#include "CImg.h"

#include <iostream>

// TODO-CNN : not very happy to leave this in a header file... :-(
using namespace neurocl;
using namespace cimg_library;

using t_digit_interval = std::pair<size_t,size_t>;

class ocr_helper
{
public:
    ocr_helper( std::shared_ptr<network_manager_interface> net_manager )
        : m_net_manager( net_manager ) {}
    virtual ~ocr_helper() {}

    void process( const CImg<float>& input );
    const CImg<float>& cropped_numbers() { return m_cropped_numbers; }

public:

    struct reco
    {
        size_t position;
        size_t value;
        float confidence;
    };

    const std::vector<reco>& recognitions() { return m_recognitions; }

    std::string reco_string()
    {
        std::string _str;
        for ( auto& _reco : m_recognitions )
            _str += std::to_string( _reco.value );
        return _str;
    }

private:

    std::vector<reco> m_recognitions;
    CImg<float> m_cropped_numbers;
    std::shared_ptr<network_manager_interface> m_net_manager;
};

#endif //OCR_H
