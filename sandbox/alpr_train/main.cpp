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

#include "CImg.h"

#include <boost/assign/list_of.hpp>
#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>

using namespace std;
using namespace boost;
using namespace boost::assign;
using namespace boost::filesystem;

#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>

const vector<std::string> v_numbers_order =
    list_of ("0")("1")("2")("3")("4")("5")("6")("7")("8")("9");

const vector<std::string> v_letters_order =
    list_of ("A")("B")("C")("D")("E")("F")("G")("H")("I")("J")("K")("L")("M")
            ("N")("O")("P")("Q")("R")("S")("T")("U")("V")("W")("X")("Y")("Z");

const vector<std::string> v_alphanum_order =
    list_of ("A")("B")("C")("D")("E")("F")("G")("H")("I")("J")("K")("L")("M")
            ("N")("O")("P")("Q")("R")("S")("T")("U")("V")("W")("X")("Y")("Z")
            ("0")("1")("2")("3")("4")("5")("6")("7")("8")("9");

class alphanum_output
{
public:
    alphanum_output( const vector<std::string>& _order ) : m_order( _order )
    {
    }
    const std::string output( const std::string& c )
    {
        vector<std::string>::const_iterator iter = std::find( m_order.begin(), m_order.end(), c );
        size_t _index = std::distance( m_order.begin(), iter );

        std::stringstream ss;
        for( const auto& _c : m_order )
        {
            ss << ( ( _c == m_order[_index] ) ? "1" : "0" ) << " ";
        }
        return ss.str();
    }
private:
    const vector<std::string>& m_order;
};

int _main1( int argc, char *argv[] )
{
    path p( "/Users/albertmurienne/Public/license_plate" );

    std::string out_file( "alpr-train.txt" );

    if ( exists( out_file ) )
        remove( out_file );

    ofstream out( out_file );

    if( is_directory( p ) )
    {
        std::cout << p << " is a directory containing:\n";

        typedef std::pair<std::string,std::string> string_pairs;
        std::vector<string_pairs> files;

        for( const auto& entry : boost::make_iterator_range( recursive_directory_iterator( p ), {} ) )
        {
            if ( extension( entry ) == ".png" )
                files.push_back( std::make_pair( entry.path().string(), entry.path().stem().string() ) );
        }

        std::random_shuffle( files.begin(), files.end() );

        alphanum_output ao( v_alphanum_order );

        for( const auto& _pair, files )
        {
            out << _pair.first << " " << ao.output( _pair.second ) << std::endl;
        }

        out.close();
    }
    else
        std::cout << p << " is not a directory\n";

    return 0;
}

int _main2( int argc, char *argv[] )
{
    path p_in( "/Users/albertmurienne/Public/license_plate" );
    path p_out( "/Users/albertmurienne/Public/mixed_morpho" );

    std::string out_file( "alpr-train2.txt" );

    if ( exists( out_file ) )
        remove( out_file );

    ofstream out( out_file );

    typedef std::pair<std::string,std::string> string_pairs;
    std::vector<string_pairs> files;

    int idx = 0;

    for( const auto& entry : boost::make_iterator_range( recursive_directory_iterator( p_in ), {} ) )
    {
        if ( extension( entry ) == ".png" )
        {
            cimg_library::CImg<float> _img( entry.path().string().c_str() );
            _img.dilate( 2 );

            std::string _out_stem = entry.path().stem().string() + std::to_string( idx++ ) + ".png";
            std::string _out = p_out.string() + "/" + _out_stem;

            _img.save( _out.c_str() );

            files.push_back( std::make_pair( _out, entry.path().stem().string() ) );
        }
    }

    std::random_shuffle( files.begin(), files.end() );

    alphanum_output ao( v_alphanum_order );

    for( const auto& _pair : files )
    {
        out << _pair.first << " " << ao.output( _pair.second ) << std::endl;
    }

    out.close();

    return 0;
}

int _main3( int argc, char *argv[] )
{
    path p( "/Users/albertmurienne/Public/license_plate" );

    std::string out_file_num( "alpr-train-num.txt" );
    std::string out_file_let( "alpr-train-let.txt" );

    if ( exists( out_file_num ) )
        remove( out_file_num );

    if ( exists( out_file_let ) )
        remove( out_file_let );

    ofstream out_num( out_file_num );
    ofstream out_let( out_file_let );

    if( is_directory( p ) )
    {
        std::cout << p << " is a directory containing:\n";

        typedef std::pair<std::string,std::string> string_pairs;
        std::vector<string_pairs> files;

        for( const auto& entry : boost::make_iterator_range( recursive_directory_iterator( p ), {} ) )
        {
            if ( extension( entry ) == ".png" )
                files.push_back( std::make_pair( entry.path().string(), entry.path().stem().string() ) );
        }

        std::random_shuffle( files.begin(), files.end() );

        alphanum_output ao_num( v_numbers_order );
        alphanum_output ao_let( v_letters_order );

        for( const auto& _pair : files )
        {
            std::string letter = _pair.second.substr( 0, 1 );

            if ( std::find( v_numbers_order.begin(), v_numbers_order.end(), letter ) != v_numbers_order.end() )
            {
                out_num << _pair.first << " " << ao_num.output( letter ) << std::endl;
            }
            else
            {
                out_let << _pair.first << " " << ao_let.output( letter ) << std::endl;
            }
        }

        out_num.close();
        out_let.close();
    }
    else
        std::cout << p << " is not a directory\n";

    return 0;
}

int main( int argc, char *argv[] )
{
    return _main3( argc, argv );
}
