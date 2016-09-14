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

#include "thebrain.h"

thebrain::thebrain() : m_current_compute_range( 0 ), m_current_face_type( 0 ), m_bStop( false )
{
    m_speech_manager.speak( "Welcome in NeuroPiCam" );
    m_thread = std::make_shared<boost::thread>( boost::bind( &thebrain::_run, this ) );
}

thebrain::~thebrain()
{
	m_bStop = true;
	m_thread->join();
}

void thebrain::push_face_type( int type )
{
    m_queue.push( type );
}

void thebrain::_run()
{
	auto avg_func = [this] ( int type ) { m_current_face_type += type; };

	while( !m_bStop )
	{
		// compute average face value
		m_current_compute_range += m_queue.consume_all( avg_func );

		if ( m_current_compute_range >= BRAIN_QUEUE_SIZE )
		{
			m_current_face_type = static_cast<int>( 0.5f + 
				( static_cast<float>( m_current_face_type ) / m_current_compute_range ) );
			m_speech_manager.set_listener( static_cast<face_type>( m_current_face_type ) );
			m_current_face_type = 0;
			m_current_compute_range = 0;
		}

		boost::this_thread::sleep( boost::posix_time::milliseconds( 500 ) );
	}
}
