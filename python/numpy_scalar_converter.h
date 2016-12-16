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

#ifndef NUMPY_SCALAR_CONVERTER_H
#define NUMPY_SCALAR_CONVERTER_H

#include <boost/python.hpp>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

// http://docs.scipy.org/doc/numpy/reference/c-api.array.html#importing-the-api
#define PY_ARRAY_UNIQUE_SYMBOL printnum_cpp_module_PyArray_API
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>


/*
 * Boost python converter for numpy scalars, e.g. numpy.uint32(123).
 * Enables automatic conversion from numpy.intXX, floatXX
 * in python to C++ char, short, int, float, etc.
 * When casting from float to int (or wide int to narrow int),
 * normal C++ casting rules apply.
 *
 * Like all boost::python converters, this enables automatic conversion for function args
 * exposed via boost::python::def(), as well as values converted via boost::python::extract<>().
 *
 * Copied from the VIGRA C++ library source code (MIT license).
 * http://ukoethe.github.io/vigra
 * https://github.com/ukoethe/vigra
 */
template <typename ScalarType>
struct NumpyScalarConverter
{
    NumpyScalarConverter()
    {
        using namespace boost::python;
        converter::registry::push_back( &convertible, &construct, type_id<ScalarType>());
    }

    // Determine if obj_ptr is a supported numpy.number
    static void* convertible(PyObject* obj_ptr)
    {
        if (PyArray_IsScalar(obj_ptr, Float32) ||
            PyArray_IsScalar(obj_ptr, Float64) ||
            PyArray_IsScalar(obj_ptr, Int8)    ||
            PyArray_IsScalar(obj_ptr, Int16)   ||
            PyArray_IsScalar(obj_ptr, Int32)   ||
            PyArray_IsScalar(obj_ptr, Int64)   ||
            PyArray_IsScalar(obj_ptr, UInt8)   ||
            PyArray_IsScalar(obj_ptr, UInt16)  ||
            PyArray_IsScalar(obj_ptr, UInt32)  ||
            PyArray_IsScalar(obj_ptr, UInt64))
        {
            return obj_ptr;
        }
        return 0;
    }

    static void construct( PyObject* obj_ptr, boost::python::converter::rvalue_from_python_stage1_data* data)
    {
        using namespace boost::python;

        // Grab pointer to memory into which to construct the C++ scalar
        void* storage = ((converter::rvalue_from_python_storage<ScalarType>*) data)->storage.bytes;

        // in-place construct the new scalar value
        ScalarType * scalar = new (storage) ScalarType;

        if (PyArray_IsScalar(obj_ptr, Float32))
            (*scalar) = PyArrayScalar_VAL(obj_ptr, Float32);
        else if (PyArray_IsScalar(obj_ptr, Float64))
            (*scalar) = PyArrayScalar_VAL(obj_ptr, Float64);
        else if (PyArray_IsScalar(obj_ptr, Int8))
            (*scalar) = PyArrayScalar_VAL(obj_ptr, Int8);
        else if (PyArray_IsScalar(obj_ptr, Int16))
            (*scalar) = PyArrayScalar_VAL(obj_ptr, Int16);
        else if (PyArray_IsScalar(obj_ptr, Int32))
            (*scalar) = PyArrayScalar_VAL(obj_ptr, Int32);
        else if (PyArray_IsScalar(obj_ptr, Int64))
            (*scalar) = PyArrayScalar_VAL(obj_ptr, Int64);
        else if (PyArray_IsScalar(obj_ptr, UInt8))
            (*scalar) = PyArrayScalar_VAL(obj_ptr, UInt8);
        else if (PyArray_IsScalar(obj_ptr, UInt16))
            (*scalar) = PyArrayScalar_VAL(obj_ptr, UInt16);
        else if (PyArray_IsScalar(obj_ptr, UInt32))
            (*scalar) = PyArrayScalar_VAL(obj_ptr, UInt32);
        else if (PyArray_IsScalar(obj_ptr, UInt64))
            (*scalar) = PyArrayScalar_VAL(obj_ptr, UInt64);

        // Stash the memory chunk pointer for later use by boost.python
        data->convertible = storage;
    }
};

#endif //NUMPY_SCALAR_CONVERTER_H
