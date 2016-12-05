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

#ifndef TENSOR_TANK_H
#define TENSOR_TANK_H

namespace neurocl { namespace convnet {

class tensor_tank
{
public:
    static tensor_tank& instance() { static tensor_tank tt; return tt; }

    tensor& get( const size_t width, const size_t height, const size_t depth1, const size_t depth2 )
    {
        const std::string key = boost::str( boost::format{"%1%x%2%x%3%x%4%"} % width % height % depth1 % depth2 );

        auto iter = m_tensor_tank.find( key );
        if ( iter != m_tensor_tank.end() )
            return m_tensor_tank.at( key );
        else
        {
            m_tensor_tank.emplace( key, tensor{} );
            tensor& t = m_tensor_tank.at( key );
            t.resize( width, height, depth1, depth2 );
            return t;
        }
    }

private:
    tensor_tank() {}
    virtual ~tensor_tank() {}

private:
    std::map<std::string,tensor> m_tensor_tank;
};

} /*namespace neurocl*/ } /*namespace convnet*/

#endif //TENSOR_TANK_H
