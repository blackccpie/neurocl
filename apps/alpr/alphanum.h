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

#ifndef ALPHANUM_H
#define ALPHANUM_H

#include <boost/foreach.hpp>

#include <algorithm>
#include <sstream>
#include <vector>

const std::vector<std::string> v_alphanum_order {
    "A","B","C","D","E","F","G","H","I","J","K","L","M",
    "N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
    "0","1","2","3","4","5","6","7","8","9"
};

const std::vector<std::string> v_numbers_order {
    "0","1","2","3","4","5","6","7","8","9"
};

const std::vector<std::string> v_letters_order {
    "A","B","C","D","E","F","G","H","I","J","K","L","M"
    "N","O","P","Q","R","S","T","U","V","W","X","Y","Z"
};

const std::vector<std::string> v_separators_order {"-"};

class alphanum
{
public:
    enum class data_type
    {
        NUMBER = 0,
        LETTER,
        BOTH,
        SEPARATOR,
        UNKNOWN
    };
public:
	alphanum( const size_t index, const data_type type ) : m_index( index )
    {
        _init_type( type );
    }
    alphanum( const std::string& c, const data_type type )
    {
        _init_type( type );

        std::vector<std::string>::const_iterator iter = std::find( m_order->begin(), m_order->end(), c );
        m_index = std::distance( m_order->begin(), iter );
    }
    const std::string string()
    {
        return (*m_order)[m_index];
    }
    const std::string bitset_string()
    {
        std::stringstream ss;
        BOOST_FOREACH( const std::string& _c, *m_order )
        {
            ss << ( ( _c == (*m_order)[m_index] ) ? "1" : "0" ) << " ";
        }
        return ss.str();
    }
private:
    void _init_type( const data_type type )
    {
        switch(type)
        {
        case data_type::NUMBER:
            m_order = &v_numbers_order;
            break;
        case data_type::LETTER:
            m_order = &v_letters_order;
            break;
        case data_type::SEPARATOR:
            m_order = &v_separators_order;
            break;
        case data_type::BOTH:
        case data_type::UNKNOWN:
        default:
            m_order = &v_alphanum_order;
            break;
        }
    }
private:
    size_t m_index;
    const std::vector<std::string>* m_order;
};

#endif //ALPHANUM_H
