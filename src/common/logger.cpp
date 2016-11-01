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

#include <iostream>
#include <iomanip>
#include <fstream>

// Implementation which allows to write into cout
class cout_log_policy : public log_policy_interface
{
public:
	cout_log_policy() {}
    virtual ~cout_log_policy() {}
	void open_ostream( const std::string& name ) override {}
	void close_ostream() override {}
	void write( const std::string& msg ) override
		{ std::cout << msg; }

private:
    std::unique_ptr<std::ofstream> m_out_stream;
};

// Implementation which allows to write into a file
class file_log_policy : public log_policy_interface
{
public:
	file_log_policy() : m_out_stream( new std::ofstream{} )
	{
	}
    virtual ~file_log_policy()
	{
		if( m_out_stream )
			close_ostream();
	}
	void open_ostream( const std::string& name ) override
	{
		m_out_stream->open( name.c_str(), std::ios_base::binary|std::ios_base::out );
		if( !m_out_stream->is_open() )
			throw( std::runtime_error( "LOGGER: Unable to open an output stream" ) );
	}
	void close_ostream() override
	{
		if( m_out_stream )
			m_out_stream->close();
	}
	void write( const std::string& msg ) override
	{
		(*m_out_stream) << msg << std::endl;
	}

private:
    std::unique_ptr<std::ofstream> m_out_stream;
};

logger::logger( const policy_type& type, const std::string& name ) : m_name( name )
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

	if( !m_policy )
		throw std::runtime_error("LOGGER: Unable to create the logger instance");
	m_policy->open_ostream( name );
}

logger::logger( logger&& l )
{
	m_name = std::move( l.m_name );
    m_log_stream = std::move( l.m_log_stream );
	m_policy = std::move( l.m_policy );
}

logger::~logger()
{
	if( m_policy )
		m_policy->close_ostream();
}

void logger::_print_impl( const std::string& msg )
{
	m_log_stream << msg;
	m_policy->write( _get_logline_header() + m_log_stream.str() );
	m_log_stream.str( "" );
}

std::string logger::_get_time()
{
	auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t( now );

    std::stringstream ss;
    ss << std::put_time( std::localtime( &in_time_t ), "%X" );
    return ss.str();
}

std::string logger::_get_logline_header()
{
	std::stringstream header;
	header << " | " << _get_time() << " | " << m_name << " ";
	return header.str();
}
