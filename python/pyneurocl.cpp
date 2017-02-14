/*
The MIT License

Copyright (c) 2015-2017 Albert Murienne

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

#include "imagetools/ocr.h"

#include <boost/python.hpp>

#include <numpy/arrayobject.h>

#include <iostream>
#include <functional>

using namespace neurocl;

void translateException( const network_exception& e )
{
    PyErr_SetString( PyExc_UserWarning, e.what() );
}

class releaseGIL{
public:
    inline releaseGIL(){
        save_state = PyEval_SaveThread();
    }

    inline ~releaseGIL(){
        PyEval_RestoreThread(save_state);
    }
private:
    PyThreadState *save_state;
};

class py_neurocl_helper
{
public:
    py_neurocl_helper( bool verbose ) : m_progress( 0 )
    {
		logger_manager& lm = logger_manager::instance();
		m.add_logger( policy_type::cout, "pyneurocl" );
		lm.add_logger( policy_type::file, "pyneurocl.log" );

        if ( !verbose )
            std::cout.rdbuf(NULL);
    }
    virtual ~py_neurocl_helper() { if ( m_net_manager ) uninit(); }

    void init( const std::string& topology, const std::string& weights )
    {
        m_net_manager = network_factory::build();
        m_net_manager->load_network( topology, weights );
    }

    void train( const std::string& samples, const int epochs, const int batch )
    {
        // Release the Global Interpreter Lock
        // so that python bytecode can be executed concurrently during training
        // http://stackoverflow.com/questions/8009613/boost-python-not-supporting-parallelism
        releaseGIL unlock = releaseGIL();

        m_progress = 0;

        m_smp_manager.load_samples( samples );
        m_net_manager->batch_train( m_smp_manager, epochs, batch, std::bind( &py_neurocl_helper::_progress, this, std::placeholders::_1 ) );
    }

    int train_progress()
    {
        //releaseGIL unlock = releaseGIL();

        return m_progress;
    }

    void compute( const boost::python::numeric::array& in, boost::python::numeric::array& out )
    {
        using namespace boost::python;

		const tuple &shape_in = extract<tuple>( in.attr("shape") );
		const tuple &shape_out = extract<tuple>( out.attr("shape") );

        int wi = extract<int>( shape_in[1] ); // cols
        int hi = extract<int>( shape_in[0] ); // rows

        int wo = extract<int>( shape_out[1] );
        int ho = extract<int>( shape_out[0] );

		std::cout << wi << " " << hi << " " << wo << " " << ho << std::endl;

        boost::shared_array<float> _in( new float[wi*hi] );
        boost::shared_array<float> _out( new float[wo*ho] );

		_array_converter<unsigned char,float>( in, _in.get() );

        sample _sample( wi * hi, _in.get() , wo * ho, _out.get() );

        m_net_manager->compute_output( _sample );

        for ( size_t i=0; i<wo*ho; i++ )
            out[i] = _out[i];
    }

    void uninit()
    {
        m_net_manager->save_network();
        m_net_manager.reset();
    }

    /********** ADVANCED FEATURES ***************/

    std::string digit_recognizer( const boost::python::numeric::array& in )
    {
        using namespace boost::python;

		const tuple &shape_in = extract<tuple>( in.attr("shape") );

        int wi = extract<int>( shape_in[1] ); // cols
        int hi = extract<int>( shape_in[0] ); // rows

        boost::shared_array<float> input( new float[wi*hi] );

        std::cout << "digit reco input image is " << wi << "x" << hi << std::endl;

        _array_converter<unsigned char,float>( in, input.get() );

        ocr_helper helper( m_net_manager );
        helper.process( input.get(), wi, hi );

        return helper.reco_string();
    }

private:

    void _progress( int p )
    {
        m_progress = p;
    }

	template <typename Ti,typename To>
	void _array_converter( const boost::python::numeric::array& in, To* out )
    {
        using namespace boost::python;

        const tuple &shape_in = extract<tuple>( in.attr("shape") );

        int wi = extract<int>( shape_in[1] ); // cols
        int hi = extract<int>( shape_in[0] ); // rows

		//std::cout << wi << " " << hi << " " << std::endl;

        PyArrayObject* np = reinterpret_cast<PyArrayObject*>( in.ptr() );
        Ti* data = reinterpret_cast<Ti*>( PyArray_DATA( np ) );

		std::copy( data, data + (wi*hi), out );

        //for ( size_t i=0; i<wi*hi; i++ )
        //    std::cout << out[i] << std::endl;
    }

private:

    int m_progress;

    samples_manager m_smp_manager;
	std::shared_ptr<network_manager_interface> m_net_manager;
};

BOOST_PYTHON_MODULE(pyneurocl)
{
	using namespace boost::python;

	boost::python::numeric::array::set_module_and_type("numpy", "ndarray");

	register_exception_translator<network_exception>( translateException );

	class_<py_neurocl_helper>("helper",init<bool>())
		.def("init",&py_neurocl_helper::init)
		.def("uninit",&py_neurocl_helper::uninit)
		.def("train",&py_neurocl_helper::train)
		.def("compute",&py_neurocl_helper::compute)
		.def("train_progress",&py_neurocl_helper::train_progress)
        .def("digit_recognizer",&py_neurocl_helper::digit_recognizer)
	;
}
