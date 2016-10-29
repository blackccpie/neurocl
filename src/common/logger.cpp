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

#include "logger.h"

file_log_policy::file_log_policy() : m_out_stream( new std::ofstream{} )
{
}

file_log_policy::~file_log_policy()
{
	if( m_out_stream )
		close_ostream();
}

void file_log_policy::open_ostream( const std::string& name )
{
    m_out_stream->open( name.c_str(), std::ios_base::binary|std::ios_base::out );
	if( !m_out_stream->is_open() )
		throw( std::runtime_error( "LOGGER: Unable to open an output stream" ) );
}

void file_log_policy::close_ostream()
{
	if( m_out_stream )
		m_out_stream->close();
}

void file_log_policy::write( const std::string& msg )
{
	(*m_out_stream) << msg << std::endl;
}

logger::logger( const policy_type& type, const std::string& name )
{
	switch( type )
	{
		case policy_type::file:
		{
			std::unique_ptr<file_log_policy> flp( new file_log_policy{} );
			m_policy = std::move( flp );
			break;
		}
		case policy_type::cout:
		default:
		{
			std::unique_ptr<cout_log_policy> clp( new cout_log_policy{} );
			m_policy = std::move( clp );
			break;
		}
			break;
	}

	m_log_line_number = 0;
	if( !m_policy )
		throw std::runtime_error("LOGGER: Unable to create the logger instance");
	m_policy->open_ostream( name );
}

logger::logger( logger&& l )
{
	m_log_line_number = std::move( l.m_log_line_number );
    m_log_stream = std::move( l.m_log_stream );
	m_policy = std::move( l.m_policy );
}

logger::~logger()
{
	if( m_policy )
		m_policy->close_ostream();
}

std::string logger::_get_time()
{
	std::string time_str;
	time_t raw_time;

	time( & raw_time );
	time_str = ctime( &raw_time );

	//without the newline character
	return time_str.substr( 0 , time_str.size() - 1 );
}

std::string logger::_get_logline_header()
{
	std::stringstream header;

	header.str("");
	header.fill('0');
	header.width(7);
	header << m_log_line_number++ << " < " << _get_time() << " - ";

	header.fill('0');
	header.width(7);
	header << clock() <<" > ~ ";

	return header.str();
}
