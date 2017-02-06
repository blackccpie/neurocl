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

#include "tensor_solver.h"
#include "tensor_operations.h"

#include "common/network_random.h"

#include <boost/iterator/zip_iterator.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>

namespace neurocl { namespace convnet {

inline void _assert_multiple( const tensor& t, const size_t& divider )
{
    if ( ( ( t.w() % divider ) != 0 ) || ( ( t.h() % divider ) != 0 ) )
        throw network_exception( "invalid tensor subsampling" );
}

inline void _assert_no_replication( const tensor& t )
{
    if ( t.d1() != 1 )
        throw network_exception( "operation not supported for replicated tensors" );
}

// check that t1.depth2 == t2.depth1
inline void _assert_cross_depths21( const tensor& t1, const tensor& t2 )
{
    if ( t1.d2() != t2.d1() )
        throw network_exception( "inconsistent tensor number of feature maps (t1.depth2 != t2.depth1)" );
}

// check that t1.depth2 == t2.depth2
inline void _assert_cross_depths22( const tensor& t1, const tensor& t2 )
{
    if ( t1.d2() != t2.d2() )
        throw network_exception( "inconsistent tensor number of feature maps (t1.depth2 != t2.depth2)" );
}

inline void _assert_muladd_sizes( const tensor& t1, const tensor& t2, const tensor& t3 )
{
    if ( ( t1.h() != t2.w() ) ||
        ( t1.w() != t3.w() ) ||
        ( t2.h() != t3.h() ) ||
        ( t1.d1() != t2.d1() ) ||
        ( t1.d2() != t2.d2() ) ||
        ( t1.d1() != t3.d1() ) ||
        ( t1.d2() != t3.d2() ) )
        throw network_exception( "inconsistent tensor multiply/add size" );
}

inline void _assert_multrans1_sizes( const tensor& t1, const tensor& t2 )
{
    if ( ( t1.w() != t2.w() ) ||
        ( t1.d1() != t2.d1() ) ||
        ( t1.d2() != t2.d2() ) )
        throw network_exception( "inconsistent tensor multiply/trans1 size" );
}

inline void _assert_multrans2_sizes( const tensor& t1, const tensor& t2 )
{
    if ( ( t1.h() != t2.h() ) ||
        ( t1.d1() != t2.d1() ) ||
        ( t1.d2() != t2.d2() ) )
        throw network_exception( "inconsistent tensor multiply/trans2 size" );
}

inline void _assert_same_sizes( const tensor& t1, const tensor& t2 )
{
    if ( ( t1.w() != t2.w() ) ||
        ( t1.h() != t2.h() ) ||
        ( t1.d1() != t2.d1() ) ||
        ( t1.d2() != t2.d2() ) )
        throw network_exception( "inconsistent tensor sizes" );
}

tensor tensor_operation::scale( const float& val, const tensor& input )
{
    tensor output;
    output.resize( input );

    tensor_foreach_p( input.d1(), input.d2() ) {
        output.m_tensor_array[d1][d2] = val * input.m_tensor_array[d1][d2];
    }

    return output;
}

tensor tensor_operation::minus( const float& val, const tensor& input )
{
    tensor output;
    output.resize( input );

    tensor_foreach_p( input.d1(), input.d2() ) {
        output.m_tensor_array[d1][d2] =
            boost::numeric::ublas::scalar_matrix<float>( input.w(), input.h(), val ) - input.m_tensor_array[d1][d2];
    }

    return output;
}

tensor tensor_operation::add( const tensor& inputA, const tensor& inputB )
{
    _assert_same_sizes( inputA, inputB );

    tensor output;
    output.resize( inputA );

    tensor_foreach_p( inputA.d1(), inputA.d2() ) {
        output.m_tensor_array[d1][d2] = inputA.m_tensor_array[d1][d2]
                + inputB.m_tensor_array[d1][d2];
    }

    return output;
}

tensor tensor_operation::sub( const tensor& inputA, const tensor& inputB )
{
    _assert_same_sizes( inputA, inputB );

    tensor output;
    output.resize( inputA );

    tensor_foreach_p( inputA.d1(), inputA.d2() ) {
        output.m_tensor_array[d1][d2] = inputA.m_tensor_array[d1][d2]
                - inputB.m_tensor_array[d1][d2];
    }

    return output;
}

tensor tensor_operation::group( const tensor& input )
{
    _assert_no_replication( input );

    tensor output;
    output.resize( input.d2() * input.w() * input.h(), 1, 1, 1 );

    auto output_mat_iter = output.m_tensor_array[0][0].data().begin();

    size_t _offset = 0;

	// TODO-CNN : could be written in a smarter way
	// ie using tensor iterators and C++11

    tensor_foreach_p( 1, input.d2() ) {

        auto input_mat_iter = input.m_tensor_array[d1][d2].data().begin();
        const size_t _size = input.m_tensor_array[d1][d2].size1() * input.m_tensor_array[d1][d2].size2();

        std::copy( input_mat_iter, input_mat_iter + _size, output_mat_iter + _offset );

        _offset += _size;
    }

    return output;
}

void tensor_operation::ungroup( const tensor& input, tensor& output )
{
    _assert_no_replication( output );

    auto input_mat_iter = input.m_tensor_array[0][0].data().begin();

    size_t _offset = 0;

	// TODO-CNN : could be written in a smarter way
	// ie using tensor iterators and C++11

    tensor_foreach_p( 1, output.d2() ) {

        const size_t _size = output.m_tensor_array[d1][d2].size1() * output.m_tensor_array[d1][d2].size2();

        std::copy( input_mat_iter, input_mat_iter + _size, output.m_tensor_array[d1][d2].data().begin() );

        input_mat_iter += _size;
    }
}

tensor tensor_operation::elediv( const tensor& inputA, const tensor& inputB )
{
    using namespace boost::numeric::ublas;

    _assert_same_sizes( inputA, inputB );

    tensor output;
    output.resize( inputA );

    tensor_foreach_p( inputA.d1(), inputA.d2() ) {
        output.m_tensor_array[d1][d2] =
            element_div( inputA.m_tensor_array[d1][d2], inputB.m_tensor_array[d1][d2] );
    }

    return output;
}

tensor tensor_operation::elemul( const tensor& inputA, const tensor& inputB )
{
    using namespace boost::numeric::ublas;

    _assert_same_sizes( inputA, inputB );

    tensor output;
    output.resize( inputA );

    tensor_foreach_p( inputA.d1(), inputA.d2() ) {
        output.m_tensor_array[d1][d2] =
            element_prod( inputA.m_tensor_array[d1][d2], inputB.m_tensor_array[d1][d2] );
    }

    return output;
}

tensor tensor_operation::mul( const tensor& inputA, const tensor& inputB )
{
    using namespace boost::numeric::ublas;

    _assert_same_sizes( inputA, inputB );

    tensor output;
    output.resize( inputA );

    tensor_foreach_p( inputA.d1(), inputA.d2() ) {
        output.m_tensor_array[d1][d2] =
            prod( inputA.m_tensor_array[d1][d2], inputB.m_tensor_array[d1][d2] );
    }

    return output;
}

tensor tensor_operation::muladd( const tensor& inputA, const tensor& inputB, const tensor& inputC )
{
    using namespace boost::numeric::ublas;

    _assert_muladd_sizes( inputA, inputB, inputC );

    tensor output;
    output.resize( inputC ); // output is homogenous to inputC

    tensor_foreach_p( inputA.d1(), inputA.d2() ) {
        output.m_tensor_array[d1][d2] = prod( inputA.m_tensor_array[d1][d2], inputB.m_tensor_array[d1][d2] )
                + inputC.m_tensor_array[d1][d2];
    }

    return output;
}

tensor tensor_operation::multrans1( const tensor& inputA, const tensor& inputB )
{
    using namespace boost::numeric::ublas;

    _assert_multrans1_sizes( inputA, inputB );

    tensor output;
    output.resize( inputA.h(), inputB.h(), inputA.d1(), inputA.d2() );

    tensor_foreach_p( inputA.d1(), inputA.d2() ) {
        output.m_tensor_array[d1][d2] =
            prod( trans( inputA.m_tensor_array[d1][d2] ), inputB.m_tensor_array[d1][d2] );
    }

    return output;
}

tensor tensor_operation::multrans2( const tensor& inputA, const tensor& inputB )
{
    using namespace boost::numeric::ublas;

    _assert_multrans2_sizes( inputA, inputB );

    tensor output;
    output.resize( inputA.w(), inputB.w(), inputA.d1(), inputA.d2() );

    tensor_foreach_p( inputA.d1(), inputA.d2() ) {
        output.m_tensor_array[d1][d2] =
            prod( inputA.m_tensor_array[d1][d2], trans( inputB.m_tensor_array[d1][d2] ) );
    }

    return output;
}

tensor tensor_operation::sqrt( const tensor& input )
{
    tensor output;
    output.resize( input );

    tensor_foreach_p( output.d1(), output.d2() ) {
        matrixF& _output = output.m_tensor_array[d1][d2];
        _output = input.m_tensor_array[d1][d2];
        std::transform( _output.data().begin(), _output.data().end(), _output.data().begin(), std::ptr_fun<float,float>( std::sqrt ) );
    }

    return output;
}

struct flipper
{
    flipper( int sx, int sy ) { m_flipped = matrixF(sx,sy); }
    const matrixF& flipped( const matrixF& in )
    { m_flipped = in; std::reverse( m_flipped.data().begin(), m_flipped.data().end() ); return m_flipped; }
    matrixF m_flipped;
};

// name reflects the feature maps feed forwarding specifity of this method
template <>
tensor tensor_operation::convolve_add_forward<tensor_operation::kernel_mode::flip,tensor_operation::pad_mode::valid>(
    const tensor& input, const tensor& filter, const int stride )
{
    using namespace boost::numeric::ublas;

    _assert_cross_depths21( input, filter );

    tensor output;

    // W2 = W1 - F + 1
    auto stepsX = input.w() - filter.w() + 1;
    auto stepsY = input.h() - filter.h() + 1;

    // no replication in output features
    output.resize( stepsX, stepsY, 1, filter.d2() );

    flipper f( filter.w(), filter.h() );

    // NOTE : tricky thing is that filter tensor replication level (filter.d1)
    // is equal to input tensor feature maps level (prev_layer.d2);
    // whereas filter feature maps level is equal to output tensor feature maps level

    for ( auto d2 = 0; d2 < filter.d2(); d2++ )
    {
    	for ( auto d1 = 0; d1 < filter.d1(); d1++ )
        {
            for ( auto j=0; j<stepsY; j++ )
            {
                for ( auto i=0; i<stepsX; i++ )
                {
                    // multiply
                    matrixF conv = element_prod( f.flipped( filter.m_tensor_array[d1][d2] ),
                        project( input.m_tensor_array[0][d1],
                            range( i, i+filter.w() ),
                            range( j, j+filter.h() ) ) );

                    // accumulate + add
                    output.m_tensor_array[0][d2](i,j) +=
                        std::accumulate( conv.data().begin(), conv.data().end(), 0.f );
                }
            }
        }
    }

    return output;
}

// name reflects the error back propagation specifity of this method
template <>
tensor tensor_operation::convolve_add_backward<tensor_operation::kernel_mode::std,tensor_operation::pad_mode::full>(
    const tensor& input, const tensor& filter, const int stride )
{
    using namespace boost::numeric::ublas;

    _assert_cross_depths22( input, filter );

    auto _FmX = filter.w() - 1;
    auto _FmY = filter.h() - 1;

    // W1 = W2 + F - 1
    auto stepsX = input.w() + _FmX;
    auto stepsY = input.h() + _FmY;

    tensor output;
    output.resize( stepsX, stepsY, 1, filter.d1() );

    // W3 = W2 + 2F - 2
    auto padX = stepsX + _FmX;
    auto padY = stepsY + _FmY;

    // TODO-CNN : could be optimized with a tensor_tank::instance().get( padX, padY, 1, filter.d2() )
    tensor padded_input;
    padded_input.resize( padX, padY, 1, filter.d2() );

    for ( auto d2 = 0; d2 < filter.d2(); d2++ )
    {
        // update padded matrix
        project(    padded_input.m_tensor_array[0][d2],
                    range( _FmX, _FmX + input.w() ),
                    range( _FmY, _FmY + input.h() ) )
            = input.m_tensor_array[0][d2];

        for ( auto d1 = 0; d1 < filter.d1(); d1++ )
        {
            for ( auto j=0; j<stepsY; j++ )
            {
                for ( auto i=0; i<stepsX; i++ )
                {
					// multiply
                    matrixF conv = element_prod( filter.m_tensor_array[d1][d2],
                        project( padded_input.m_tensor_array[0][d2],
                            range( i, i+filter.w() ),
                            range( j, j+filter.h() ) ) );

                    // accumulate + add
                    // question was raised about the necessity to divide proportionnaly to forward feed accumulation filter replication
                    output.m_tensor_array[0][d1](i,j) +=
                        std::accumulate( conv.data().begin(), conv.data().end(), 0.f );
                }
            }
        }
    }

    return output;
}

// name reflects the filters gradient update specifity of this method
template <>
tensor tensor_operation::convolve_update<tensor_operation::kernel_mode::std,tensor_operation::pad_mode::valid>(
    const tensor& input, const tensor& filter, const int stride )
{
    using namespace boost::numeric::ublas;

    tensor output;

    // W2 = W1 - F + 1
    auto stepsX = input.w() - filter.w() + 1;
    auto stepsY = input.h() - filter.h() + 1;

    // no replication in output features
    output.resize( stepsX, stepsY, input.d2(), filter.d2() );

    for ( auto d1 = 0; d1 < input.d2(); d1++ )
    {
        for ( auto d2 = 0; d2 < filter.d2(); d2++ )
        {
            for ( auto j=0; j<stepsY; j++ )
            {
                for ( auto i=0; i<stepsX; i++ )
                {
					// multiply
                    matrixF conv = element_prod( filter.m_tensor_array[0][d2],
                        project( input.m_tensor_array[0][d1],
                            range( i, i+filter.w() ),
                            range( j, j+filter.h() ) ) );

                    // accumulate
                    output.m_tensor_array[d1][d2](i,j) =
                        std::accumulate( conv.data().begin(), conv.data().end(), 0.f );
                }
            }
        }
    }

    return output;
}

tensor tensor_operation::subsample( const tensor& input, const size_t subsample )
{
    _assert_multiple( input, subsample );

    tensor output;
    output.resize( input.w() / subsample, input.h() / subsample, input.d1(), input.d2() );

    tensor_foreach_p( input.d1(), input.d2() )
    {
        matrixF& feature_map = output.m_tensor_array[d1][d2];

        const matrixF& prev_feature_map = input.m_tensor_array[d1][d2];
        auto prev_width = prev_feature_map.size1();
        auto prev_it1 = prev_feature_map.begin1();

        for( auto it1 = feature_map.begin1(); it1 != feature_map.end1(); it1++, prev_it1 += subsample )
        {
            auto prev_it2 = prev_it1.begin();
            for( auto it2 = it1.begin(); it2 != it1.end(); it2++, prev_it2 += subsample )
            {
                float max_value = std::numeric_limits<float_t>::lowest();

                // could use ublas::project + std::accumulate + std::max for more compact expression

                // compute max in subsampling zone
                for ( auto j =0; j<subsample; j++ )
                	for ( auto i =0; i<subsample; i++ )
                    {
                        const float& value = *(prev_it2 + i + (j*prev_width) );
                        if ( value > max_value )
                            max_value = value;
                    }

                // update value in the destination feature map
                *it2 = max_value;
            }
        }
    }

    return output;
}

tensor tensor_operation::d_subsample( const tensor& input, const tensor& input_ref, const size_t subsample )
{
    _assert_multiple( input_ref, subsample );

    tensor output;
    output.resize( input.w() * subsample, input.h() * subsample, input.d1(), input.d2() );

    tensor_foreach_p( input.d1(), input.d2() )
    {
        // Initialize iterators
        const matrixF& prev_feature_map = input_ref.m_tensor_array[d1][d2];
        auto prev_width = prev_feature_map.size1();

        const matrixF& error_map = input.m_tensor_array[d1][d2];
        auto err_it1 = error_map.begin1();

        matrixF& prev_error_map = output.m_tensor_array[d1][d2];
        auto prev_err_iter1 = prev_error_map.begin1();

        // Iterate
        for( auto prev_it1 = prev_feature_map.begin1(); prev_it1 != prev_feature_map.end1();
            prev_it1 += subsample, prev_err_iter1 += subsample, ++err_it1 )
        {
            auto prev_err_it2 = prev_err_iter1.begin();
            auto err_it2 = err_it1.begin();
            for( auto prev_it2 = prev_it1.begin(); prev_it2 != prev_it1.end();
                prev_it2 += subsample, prev_err_it2 += subsample, ++err_it2 )
            {
                float max_value = std::numeric_limits<float_t>::lowest();

                int max_offset = 0;

                // compute max in subsampling zone
                for ( auto j =0; j<subsample; j++ )
                	for ( auto i =0; i<subsample; i++ )
                    {
                        const float& value = *(prev_it2 + i + (j*prev_width) );
                        if ( value > max_value )
                        {
                            max_value = value;
                            max_offset = i + (j*prev_width);
                        }
                    }

                // update error on last layer max value pixel
                *(prev_err_it2 + max_offset) = *err_it2;
            }
        }
    }

    return output;
}

tensor tensor_operation::uniform_sum( const tensor& input )
{
    tensor output;
    output.resize( input );

    tensor_foreach_p( input.d1(), input.d2() ) {
        float _acc = std::accumulate( input.m_tensor_array[d1][d2].data().begin(), input.m_tensor_array[d1][d2].data().end(), 0.f );
        output.m_tensor_array[d1][d2] = boost::numeric::ublas::scalar_matrix<float>( input.w(), input.h(), _acc );
    }

    return output;
}

void tensor_operation::bernoulli( tensor& input, const float p )
{
    random::rand_bernoulli_generator bernoulli( p );

    tensor_foreach_p( input.d1(), input.d2() ) {
        std::for_each(  input.m_tensor_array[d1][d2].data().begin(),
                        input.m_tensor_array[d1][d2].data().end(),
                        [&bernoulli]( float& a ) { a = bernoulli.gen<float>(); } );
    }
}

tensor tensor_operation::binary_operator( const tensor& inputA, const tensor& inputB, std::function<float (const float&,const float&)> op )
{
    _assert_same_sizes( inputA, inputB );

    tensor output;
    output.resize( inputA );

    tensor_foreach_p( inputA.d1(), inputA.d2() ) {
        auto iter = boost::make_zip_iterator( boost::make_tuple( inputA.m_tensor_array[d1][d2].data().begin(),
            inputB.m_tensor_array[d1][d2].data().begin(), output.m_tensor_array[d1][d2].data().begin() ) );
        auto end = boost::make_zip_iterator( boost::make_tuple( inputA.m_tensor_array[d1][d2].data().end(),
            inputB.m_tensor_array[d1][d2].data().end(), output.m_tensor_array[d1][d2].data().end() ) );

            for( ; iter != end ; ++iter )
            {
                const float& ia = iter->get<0>();
                const float& ib = iter->get<1>();
                float& o = iter->get<2>();

                o = op( ia, ib );
            }
    }

    return output;
}

template<>
void tensor_operation::optimize<tensor_operation::optimize_mode::redux>( const std::shared_ptr<tensor_solver_iface>& solver, tensor* input, tensor** input_cache, const tensor* deltas )
{
    solver->update_redux( *input, input_cache, *deltas );
}

template<>
void tensor_operation::optimize<tensor_operation::optimize_mode::std>( const std::shared_ptr<tensor_solver_iface>& solver, tensor* input, tensor** input_cache, const tensor* deltas )
{
    solver->update( *input, input_cache, *deltas );
}

} /*namespace neurocl*/ } /*namespace convnet*/
