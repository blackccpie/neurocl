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

#include "autothreshold.h"

#include "CImg.h"

#include <boost/shared_array.hpp>

using namespace cimg_library;

//************** WARNING : FOR NOW ONLY WORKS ON 0-256 DYNAMIC!!! **************//

template<typename T>
int default_isodata( const T* data, const int length );
int isodata( int* data, const int length );

void auto_threshold( float* input, const int sizeX, const int sizeY )
{
    CImg<float> _input( input, sizeX, sizeY, 1, 1, true /*shared*/);

    int threshold = default_isodata<cimg_ulong>( _input.get_histogram(256), 256 );
    _input.threshold( threshold );
}

// One of the many autothreshold IJ implementations:
// https://imagej.nih.gov/ij/developer/source/ij/process/AutoThresholder.java.html

template<typename T>
int default_isodata( const T* data, const int length )
{
    int n = length;
    boost::shared_array<int> data2( new int[n] );
    int mode=0, maxCount=0;
    for (int i=0; i<n; i++) {
        data2[i] = std::round<int>( data[i] );
        if (data2[i]>maxCount) {
            maxCount = data2[i];
            mode = i;
        }
    }
    int maxCount2 = 0;
    for (int i = 0; i<n; i++) {
        if ((data2[i]>maxCount2) && (i!=mode))
            maxCount2 = data2[i];
    }
    int hmax = maxCount;
    if ((hmax>(maxCount2*2)) && (maxCount2!=0)) {
        hmax = (int)(maxCount2 * 1.5);
        data2[mode] = hmax;
    }
    return isodata(data2.get(),n);
}

int isodata( int* data, const int length )
{
    // This is the original ImageJ IsoData implementation, here for backward compatibility.
    int level;
    int maxValue = length - 1;
    double result, sum1, sum2, sum3, sum4;
    int count0 = data[0];
    data[0] = 0; //set to zero so erased areas aren't included
    int countMax = data[maxValue];
    data[maxValue] = 0;
    int min = 0;
    while ((data[min]==0) && (min<maxValue))
        min++;
    int max = maxValue;
    while ((data[max]==0) && (max>0))
        max--;
    if (min>=max) {
        data[0]= count0; data[maxValue]=countMax;
        level = length/2;
        return level;
    }
    int movingIndex = min;
    int inc = std::max<int>(max/40, 1);
    do {
        sum1=sum2=sum3=sum4=0.0;
        for (int i=min; i<=movingIndex; i++) {
            sum1 += (double)i*data[i];
            sum2 += data[i];
        }
        for (int i=(movingIndex+1); i<=max; i++) {
            sum3 += (double)i*data[i];
            sum4 += data[i];
        }
        result = (sum1/sum2 + sum3/sum4)/2.0;
        movingIndex++;
    } while ((movingIndex+1)<=result && movingIndex<max-1);
    data[0]= count0; data[maxValue]=countMax;
    level = std::round<int>(result);
    return level;
}
