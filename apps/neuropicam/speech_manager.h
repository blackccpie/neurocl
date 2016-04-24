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

#include "face_commons.h"

#include <string>

#include <stdlib.h>

class speech_manager
{
public:
    speech_manager() : m_current_listener( FT_UNKNOWN ) {}
    virtual ~speech_manager() {}

    void speak( const std::string& message )
    {
    #ifdef __APPLE__
        // NOT IMPLEMENTED YET
    #else
        std::string command = std::string( "cd ../../picoPi2/tts;sh speak.sh \"" )
			+ message + std::string( "\";cd -" );

        // grab using raspistill utility
        system( command.c_str() );
    #endif
    }

    void set_listener( const face_type& type )
    {
		switch( type )
		{
		case FT_USERA:
			speak( "Hello " + facecam_users::instance().nicknameA() );
			speak( "What can I do you for?" );
			break;
		case FT_USERB:
			speak( "Hello " + facecam_users::instance().nicknameA() );
			speak( "What can I do you for?" );
			break;
		case FT_UNKNOWN:
		default:
			break;
		}

		m_current_listener = type;
	}
private:
	face_type m_current_listener;
};
