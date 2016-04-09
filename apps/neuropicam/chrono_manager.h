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

#include <boost/chrono.hpp>

#include <string>

class chrono_manager
{
public:
    chrono_manager() {}
    virtual ~chrono_manager() {}

	void start()
	{
		m_labelled_durations.clear();
		m_key_time = bc::system_clock::now();
	}

	void step( const std::string& label )
	{
		boost::chrono::milliseconds duration = boost::chrono::duration_cast<bc::milliseconds>( bc::system_clock::now() - m_key_time );
		m_labelled_durations.push_back( std::make_pair( label, duration.count() ) );
		m_key_time = bc::system_clock::now();
	}

	const std::string summary()
	{
		std::string summary = "";
		typedef std::pair<std::string,int> labelled_ms_t;
		BOOST_FOREACH( const labelled_ms_t& lms, m_labelled_durations )
		{
			summary += "|";
			summary += lms.first;
			summary += "=";
			summary += boost::lexical_cast<std::string>( lms.second );
			summary += "ms|"
		}
		return summary;
	}

private:

	boost::chrono::system_clock::time_point m_key_time;
	std::vector< std::pair<std::string,int> > m_labelled_durations;
};
