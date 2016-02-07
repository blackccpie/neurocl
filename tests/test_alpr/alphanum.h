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

#include <boost/foreach.hpp>
#include <boost/assign/list_of.hpp>
using namespace boost::assign;;

#include <sstream>
#include <vector>

const std::vector<std::string> v_alphanum_order =
    list_of ("A")("B")("C")("D")("E")("F")("G")("H")("I")("J")("K")("L")("M")
            ("N")("O")("P")("Q")("R")("S")("T")("U")("V")("W")("X")("Y")("Z")
            ("0")("1")("2")("3")("4")("5")("6")("7")("8")("9");

class alphanum
{
public:
    alphanum( const size_t index ) : m_index( index ) {}
    alphanum( const std::string& c )
    {
        std::vector<std::string>::const_iterator iter = std::find( v_alphanum_order.begin(), v_alphanum_order.end(), c );
        m_index = std::distance( v_alphanum_order.begin(), iter );
    }
    const std::string string()
    {
        return v_alphanum_order[m_index];
    }
    const std::string bitset_string()
    {
        std::stringstream ss;
        BOOST_FOREACH( const std::string& _c, v_alphanum_order )
        {
            ss << ( ( _c == v_alphanum_order[m_index] ) ? "1" : "0" ) << " ";
        }
        return ss.str();
    }
private:
    size_t m_index;
};
