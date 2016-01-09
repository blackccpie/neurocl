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

#ifndef NETWORK_FILE_HANDLER_H
#define NETWORK_FILE_HANDLER_H

#include "network_interface.h"

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
namespace bfs = boost::filesystem;

namespace neurocl {

class network_file_handler
{
private:

    /*struct binary_layer
    {
        size_t size;
        float* weights;
    };*/

public:

    network_file_handler( boost::shared_ptr<network_interface> net ) : m_net( net ) {}
    virtual ~network_file_handler() {}

    void load_network_topology( const std::string& topology_path )
    {
        if ( !bfs::exists( topology_path ) )
        {
            std::cerr << "network_file_handler::load_network_topology - topology file \'" << topology_path << "\' doesn't exist" << std::endl;
            throw network_exception( "error reading topology config file" );
        }

        std::ifstream topology( topology_path );
        if ( !topology || !topology.is_open() )
        {
            std::cerr << "network_file_handler::load_network_topology - error opening topology file \'" << topology_path << "\'" << std::endl;
            throw "error opening topology config file";
        }

        std::vector<layer_size> layer_sizes;

        size_t cur_line = 0;
        size_t idx_layer = 0;
        std::string line;
        while ( std::getline( topology, line ) )
        {
            ++cur_line;

            if ( boost::starts_with( line, "layer" ) )
            {
                typedef std::vector<std::string> split_vector_type;

                split_vector_type split_vec;
                boost::split( split_vec, line, boost::is_any_of(":x") );

                if ( split_vec.size() != 4 )
                {
                    std::cerr << "network_file_handler::load_network_topology - line " << cur_line << " is malformed (missing elements)" << std::endl;
                    throw network_exception( "malformed line in topology file" );
                }

                try
                {
                    int _idx = boost::lexical_cast<int>( split_vec[1] );
                    int _x = boost::lexical_cast<int>( split_vec[2] );
                    int _y = boost::lexical_cast<int>( split_vec[3] );
                    // for now index are supposed to be declared in increasing order...
                    if ( _idx != idx_layer )
                    {
                        std::cerr << "network_file_handler::load_network_topology - line " << cur_line << " is malformed (wrong layer index)" << std::endl;
                        throw network_exception( "malformed line in topology file" );
                    }

                    std::cout << "network_file_handler::load_network_topology - adding layer" << _idx << " of size " << _x << "x" << _y << std::endl;

                    layer_sizes.push_back( neurocl::layer_size( _x, _y ) );
                }
                catch( network_exception& )
                {
                    // re-throw network_exception
                    throw;
                }
                catch(...)
                {
                    std::cerr << "network_file_handler::load_network_topology - line " << cur_line << " is malformed (an element is not a number)" << std::endl;
                    throw network_exception( "malformed line in topology file" );
                }

                ++idx_layer;
            }
        }

        if ( !layer_sizes.empty() )
            m_net->add_layers_2d( layer_sizes );
        else
            throw network_exception( "empty topology file" );
    }

    void load_network_weights( const std::string& weights_path )
    {
        // NOT IMPLEMENTED YET
        return;

        std::ifstream network_file( weights_path, std::ios::in|std::ios::binary|std::ios::ate );

        if ( network_file.is_open() )
        {
            // std::streampos size = network_file.tellg();
            // char* memblock = new char[size];
            // file.seekg (0, ios::beg);
            // file.read (memblock, size);
            // file.close();
            //delete[] memblock;
          }
          else
            throw network_exception( "unable to open network file" );
    }

    void save_weights()
    {
    }

private:

    std::string m_weights_path;

    boost::shared_ptr<network_interface> m_net;
};

} //namespace neurocl

#endif //NETWORK_FILE_HANDLER_H
