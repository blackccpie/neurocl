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

#include "neurocl.h"

#include <boost/python.hpp>

using namespace neurocl;

void translateException( const network_exception& e )
{
    PyErr_SetString( PyExc_UserWarning, e.what() );
}

class py_neurocl_helper
{
public:
    py_neurocl_helper() : m_smp_manager( neurocl::samples_manager::instance() ) {}
    virtual ~py_neurocl_helper() { uninit(); }

    void init( const std::string& topology, const std::string& weights )
    {
        m_net_manager = network_factory::build();
        m_net_manager->load_network( topology, weights );
    }

    void train( const std::string& samples, const int epochs, const int batch )
    {
        m_smp_manager.load_samples( samples );
        m_net_manager->batch_train( m_smp_manager, epochs, batch );
    }

    void uninit()
    {
        m_net_manager->save_network();
        m_net_manager.reset();
    }
private:
    samples_manager& m_smp_manager;
    std::shared_ptr<network_manager_interface> m_net_manager;
};

BOOST_PYTHON_MODULE(pyneurocl)
{
  using namespace boost::python;

  register_exception_translator<network_exception>( translateException );

  class_<py_neurocl_helper>("helper")
    .def("init",&py_neurocl_helper::init)
    .def("uninit",&py_neurocl_helper::uninit)
    .def("train",&py_neurocl_helper::train)
  ;
}
