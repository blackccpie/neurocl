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

#ifndef LOGGER_H
#define LOGGER_H

#include <string>
#include <memory>
#include <mutex>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

//static logging::logger<logging::file_log_policy> log_inst( "execution.log" );

#define LOG log_inst.print< logging::severity_type::debug >
#define LOG_ERR log_inst.print< logging::severity_type::error >
#define LOG_WARN log_inst.print< logging::severity_type::warning >

class log_policy_interface
{
public:
	virtual void open_ostream( const std::string& name ) = 0;
	virtual void close_ostream() = 0;
	virtual void write( const std::string& msg ) = 0;
};

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
	file_log_policy();
    virtual ~file_log_policy();
	void open_ostream( const std::string& name ) override;
	void close_ostream() override;
	void write( const std::string& msg ) override;

private:
    std::unique_ptr<std::ofstream> m_out_stream;
};

enum class severity_type
{
	debug = 1,
	error,
	warning
};

enum class policy_type
{
	cout = 1,
	file
};

class logger
{
public:
	logger( const policy_type& type, const std::string& name );
	logger( logger&& l );
    virtual ~logger();

	template<severity_type severity,typename...Args>
	void print( const std::string& msg )
	{
		m_write_mutex.lock();
		switch( severity )
		{
			case severity_type::debug:
				m_log_stream << "| D |";
				break;
			case severity_type::warning:
				m_log_stream << "| W |";
				break;
			case severity_type::error:
				m_log_stream << "| E |";
				break;
		};
		_print_impl( msg );
		m_write_mutex.unlock();
	}

private:
    std::string _get_time();
	std::string _get_logline_header();

	//Core printing functionality
	void _print_impl( const std::string& msg )
	{
		m_log_stream << msg;
		m_policy->write( _get_logline_header() + m_log_stream.str() );
		m_log_stream.str( "" );
	}

private:
	unsigned int m_log_line_number;
    std::stringstream m_log_stream;
	std::unique_ptr<log_policy_interface> m_policy;
	std::mutex m_write_mutex;
};

class logger_manager
{
public:
	static logger_manager& instance()
    {
        static logger_manager s;
        return s;
    }

	void add_logger( const policy_type& type, const std::string& name )
	{
		m_loggers.emplace_back( logger( type, name ) );
	}

	bool empty_logger()
	{
		return m_loggers.empty();
	}

	template<severity_type severity>
	void print( const std::string& msg )
	{
		for( auto& logger : m_loggers )
		{
			logger.print<severity>( msg );
		}
	}
private:
	logger_manager() {}
	virtual ~logger_manager() {}

private:
	std::vector<logger> m_loggers;
};

//! log as stream
template<severity_type severity>
class streamlogger
{
public:
	streamlogger() : m_stream( new std::ostringstream ) {};
	streamlogger(const streamlogger& copy) : m_stream( copy.m_stream ) { copy.m_stream = 0; };
	~streamlogger()
	{
		if( m_stream != 0 )
		{
			logger_manager::instance().print<severity>( m_stream->str() );
			delete m_stream;
		}
	};

	std::ostringstream* stream() const
		{ return m_stream; };
	operator bool() const
		{ return logger_manager::instance().empty_logger(); }

private:
	mutable std::ostringstream* m_stream;
};

//! helper to log a stream with level and to do nothing if the logger has no sink registered
#define LOGGER(level) if( streamlogger<severity_type::level> keep=streamlogger<severity_type::level>() ) ; else (*keep.stream())

#endif //LOGGER_H
