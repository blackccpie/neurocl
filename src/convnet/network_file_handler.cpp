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

#include "network_file_handler.h"
#include "network_interface.h"

#include "common/network_exception.h"
#include "common/logger.h"

#include "common/portable_binary_archive/portable_binary_iarchive.hpp"
#include "common/portable_binary_archive/portable_binary_oarchive.hpp"

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>
namespace bfs = boost::filesystem;

#include <fstream>

namespace neurocl { namespace convnet {

std::istream& operator>> ( std::istream &input, layer_type& type )
{
    std::string layer_string;
    input >> layer_string;

    if ( layer_string == "conv" )
        type = CONV_LAYER;
    else if ( layer_string == "pool" )
        type = POOL_LAYER;
    else if ( layer_string == "full" )
        type = FULL_LAYER;
    else if ( layer_string == "in" )
        type = INPUT_LAYER;
    else if ( layer_string == "out" )
        type = OUTPUT_LAYER;
    else
        input.setstate( std::ios_base::failbit );

    return input;
}

network_file_handler::network_file_handler( const std::shared_ptr<network_interface>& net ) : m_net( net ), m_layers( 0 )
{
}

network_file_handler::~network_file_handler()
{
}

void network_file_handler::load_network_topology( const std::string& topology_path )
{
    if ( !bfs::exists( topology_path ) )
    {
        LOGGER(error) << "network_file_handler::load_network_topology - topology file \'" << topology_path << "\' doesn't exist" << std::endl;
        throw network_exception( "error reading topology config file" );
    }

    std::ifstream topology( topology_path );
    if ( !topology || !topology.is_open() )
    {
        LOGGER(error) << "network_file_handler::load_network_topology - error opening topology file \'" << topology_path << "\'" << std::endl;
        throw "error opening topology config file";
    }

    std::vector<layer_descr> layers;
    m_layers = 0;

    size_t cur_line = 0;
    size_t idx_layer = 0;
    std::string line;
    while ( std::getline( topology, line ) )
    {
        ++cur_line;

        if ( boost::starts_with( line, "layer" ) )
        {
            using split_vector_type = std::vector<std::string>;

            split_vector_type split_vec;
            boost::split( split_vec, line, boost::is_any_of(":x") );

            try
            {
                layer_type _t = boost::lexical_cast<layer_type>( split_vec[1] );

            	bool is_other_layer = ( _t != CONV_LAYER ) && ( split_vec.size() == 6 );
                bool is_conv_layer = ( _t == CONV_LAYER ) && ( split_vec.size() == 7 ); // conv layer specifies filter size

                if ( !is_other_layer && !is_conv_layer )
            	{
                	LOGGER(error) << "network_file_handler::load_network_topology - line " << cur_line << " is malformed (missing elements)" << std::endl;
                	throw network_exception( "malformed line in topology file" );
            	}

                int _idx = boost::lexical_cast<int>( split_vec[2] );
                int _x = boost::lexical_cast<int>( split_vec[3] );
                int _y = boost::lexical_cast<int>( split_vec[4] );
                int _z = boost::lexical_cast<int>( split_vec[5] );
                int _f = is_conv_layer ? boost::lexical_cast<int>( split_vec[6] ) : 0;

                // for now index are supposed to be declared in increasing order...
                if ( _idx != idx_layer )
                {
                    LOGGER(error) << "network_file_handler::load_network_topology - line " << cur_line << " is malformed (wrong layer index)" << std::endl;
                    throw network_exception( "malformed line in topology file" );
                }

                // TODO-CNN : implement layer topology ordering constraints check??

                LOGGER(info) << "network_file_handler::load_network_topology - adding layer " << _idx << " of size " << _x << "x" << _y << "x" << _z
                    << ( ( _f != 0 ) ? ( ":" + std::to_string( _f ) ) : "" ) << std::endl;

                layers.push_back( layer_descr( _t, _x, _y, _z, _f ) );

                ++m_layers;
            }
            catch( network_exception& )
            {
                // re-throw network_exception
                throw;
            }
            catch(...)
            {
                LOGGER(error) << "network_file_handler::load_network_topology - line " << cur_line << " is malformed (an element is malformed)" << std::endl;
                throw network_exception( "malformed line in topology file" );
            }

            ++idx_layer;
        }
    }

    if ( !layers.empty() )
        m_net->add_layers( layers );
    else
        throw network_exception( "empty topology file" );
}

void network_file_handler::load_network_weights( const std::string& weights_path )
{
    if ( !m_layers )
        throw network_exception( "no network topology loaded" );

    // save weights file path for further saving
    m_weights_path = weights_path;

    if ( !bfs::exists( weights_path ) )
    {
        std::ofstream input_weights( weights_path, std::ios::out );
        input_weights << "TEMPORARY WEIGHTS FILE CONTENT BEFORE NETWORK SAVING";
        return;
    }

    std::ifstream input_weights( weights_path, std::ios::in | std::ios::binary );

    if ( input_weights.is_open() )
    {
        try
        {
        	portable_binary_iarchive ia( input_weights, endian_little );

            for ( size_t i=0; i<m_layers; i++ )
            {
                layer_storage l;
                LOGGER(info) << "network_file_handler::load_network_weights - loading layer" << i << " weights" << std::endl;
                ia >> l;

                layer_ptr lp( l.m_num_weights, l.m_weights, l.m_num_bias, l.m_bias );
                m_net->set_layer_ptr( i, lp );
            }
        }
        catch( boost::archive::archive_exception& e )
        {
            LOGGER(error) << "network_file_handler::load_network_weights - error decoding weights file : " << e.what() << std::endl;
            throw network_exception( "error decoding weights file" );
        }
        catch(...)
        {
            LOGGER(error) << "network_file_handler::load_network_weights - unknown error decoding weights file" << std::endl;
            throw network_exception( "error decoding weights file" );
        }
    }
    else
        throw network_exception( "unable to open weights file for loading" );

    LOGGER(info) << "network_file_handler::load_network_weights - successfully loaded network weights" << std::endl;
}

void network_file_handler::save_network_weights()
{
    std::ofstream output_weights( m_weights_path, std::ios::out | std::ios::binary | std::ios::trunc );

    if ( output_weights.is_open() )
    {
        portable_binary_oarchive oar( output_weights, endian_little );

        for ( size_t i=0; i<m_net->count_layers(); i++ )
        {
            LOGGER(info) << "network_file_handler::save_network_weights - saving layer" << i << " weights" << std::endl;
            layer_ptr ptr = m_net->get_layer_ptr( i );
            layer_storage l( ptr.num_weights, ptr.weights, ptr.num_bias, ptr.bias );
            oar << l;
        }
    }
    else
      throw network_exception( "unable to open weights file for saving" );
}

} /*namespace neurocl*/ } /*namespace convnet*/
