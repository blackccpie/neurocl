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

#include <boost/circular_buffer.hpp>

#include <chrono>
#include <string>
#include <vector>

class chrono_manager
{
public:
    chrono_manager() : m_frame_periods( 10 ) {}
    virtual ~chrono_manager() {}

	void start()
	{
		m_labelled_durations.clear();
		m_key_time = std::chrono::system_clock::now();
		m_last_frame_key_time = std::chrono::system_clock::now();
	}

	void step( const std::string& label )
	{
		std::chrono::milliseconds duration = std::chrono::duration_cast<std::chrono::milliseconds>( std::chrono::system_clock::now() - m_key_time );
		m_labelled_durations.push_back( std::make_pair( label, duration.count() ) );
		m_key_time = std::chrono::system_clock::now();
	}

	const std::string summary()
	{
		std::string summary = "";
		for( const auto& lms : m_labelled_durations )
		{
			summary += "|";
			summary += lms.first;
			summary += "=";
			summary += std::to_string( lms.second );
			summary += "ms|";
		}
		return summary;
	}

	void frame()
	{
		std::chrono::milliseconds duration = std::chrono::duration_cast<std::chrono::milliseconds>( std::chrono::system_clock::now() - m_last_frame_key_time );
		m_frame_periods.push_back( duration.count() );
		m_last_frame_key_time = std::chrono::system_clock::now();
	}

	float framerate()
	{
		float mean_fps = 0.f;
		for( const auto& period : m_frame_periods )
		{
			mean_fps += 1000.f / static_cast<float>( period );
		}
		mean_fps /= static_cast<float>( m_frame_periods.size() );
		return mean_fps;
	}

private:

	std::chrono::system_clock::time_point m_key_time;
	std::vector< std::pair<std::string,int> > m_labelled_durations;

	std::chrono::system_clock::time_point m_last_frame_key_time;
	boost::circular_buffer<int> m_frame_periods;
};
