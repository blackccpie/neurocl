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

#ifndef ALPR_H
#define ALPR_H

#include "network_manager.h"
#include "plate_resolution.h"

#include "CImg.h"

#include <string>

namespace alpr {

class network_manager;

// Class to manage license plate recognition
class license_plate
{
public:
    license_plate( const std::string& file_plate, neurocl::network_manager& net_num, neurocl::network_manager& net_let );
    ~license_plate();

    void analyze();

private:

    void _prepare_work_plate( cimg_library::CImg<float>& input_plate );
    void _compute_ranges();
    void _compute_distance_map();

private:

    cimg_library::CImg<float> m_work_plate;

    typedef std::pair<size_t,size_t> t_letter_interval;
    std::vector<t_letter_interval> m_letter_intervals;

    plate_resolution m_plate_resol;
};

} //namespace alpr

#endif //ALPR_H
