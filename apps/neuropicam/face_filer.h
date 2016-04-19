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

#ifndef FACE_FILER_H
#define FACE_FILER_H

#include "CImg.h"

#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>

#include <iostream>

// Class to manage face files
class face_filer
{
public:
    face_filer() : m_save_path( "../nets/facecam/faces" ), m_last_path( "" )
    {
    }
    virtual ~face_filer() {}

    void save_face( const std::string& label, cimg_library::CImg<float>& image )
    {
        using namespace boost::filesystem;

        bool saved = false;
        size_t idx = 0;
        do
        {
            path _path = m_save_path / path( label ) / path( boost::lexical_cast<std::string>( idx ) + ".png" );
            if ( !exists( _path ) )
            {
                if ( !exists( _path.parent_path() ) )
                    create_directory( _path.parent_path() );

                image.normalize( 0, 255 );
				m_last_path = _path.string();
                image.save( m_last_path.c_str() );
                saved = true;
            }
            idx++;

        } while ( !saved );
    }

	const std::string& last_path() { return m_last_path; }

private:

	std::string m_last_path;
    boost::filesystem::path m_save_path;
};

#endif //FACE_FILER_H
