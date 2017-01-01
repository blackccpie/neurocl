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

#ifndef SOLVER_H
#define SOLVER_H

#include "learning_scheduler.h"
#include "logger.h"
#include "network_exception.h"

#include <cmath>
#include <map>

namespace neurocl {

class solver_base : public std::enable_shared_from_this<solver_base>
{
public:
    solver_base() : m_normalize_grad( 1.f ) {}

    void register_for_scheduling()
    {
        learning_scheduler::instance().register_solver( shared_from_this() );
    }

    void set_size( const size_t& size )
    {
        if ( !size )
            throw network_exception( "cannot set solver size to zero" );

        m_normalize_grad = 1.f / static_cast<float>( size );
    }

    //! get cache size
    virtual size_t get_cache_size() = 0;

    //! get current learning rate
    virtual const float& get_learning_rate() = 0;
    //! set learning rate (scheduled learning)
    virtual void set_learning_rate( const float new_rate ) = 0;

    //! get parameters parsing map
    using t_parameters_map = std::map<const std::string,std::reference_wrapper<float>>;
    t_parameters_map& get_parameters_map()
    {
        return m_parameters_set;
    }

protected:

    //! parameters parsing map
    t_parameters_map m_parameters_set;

    float m_normalize_grad;
};

/* Stochastic Gradient Descent solver implementation */
class solver_sgd : public solver_base
{
public:
    solver_sgd( const float alpha, const float lambda, const float mu )
        : m_alpha( alpha ), m_lambda( lambda ), m_mu( mu ) {}
    solver_sgd( std::initializer_list<float> params_list )
        : m_alpha( 0.01f ), m_lambda( 0.00005f ), m_mu( 0.9f )
    {
        if ( params_list.size() == 3 )
    	{
        	m_alpha     = params_list.begin()[0];
        	m_lambda    = params_list.begin()[1];
        	m_mu        = params_list.begin()[2];
        }
        else
            LOGGER(warning) << "solver_sgd::solver_sgd - invalid parameters number, keeping defaults" << std::endl;
    }
    solver_sgd() : m_alpha( 0.01f ), m_lambda( 0.00005f ), m_mu( 0.9f )
    {
        m_parameters_set = t_parameters_map( // assignment workaround added for clang/OSX
        	{ {"lr",std::ref(m_alpha)}, {"wd",std::ref(m_lambda)}, {"m",std::ref(m_mu)} }
        );
    }
    virtual ~solver_sgd() {}

    template<typename T>
    void update( T& input, T** input_cache, const T& gradient )
    {
        T& input_momentum = *(input_cache[0]);
        input_momentum = ( m_mu * input_momentum ) - m_alpha * ( m_normalize_grad * gradient + m_lambda * input );
        input += input_momentum;
    }

    template<typename T>
    void update_redux( T& input, T** input_cache, const T& gradient )
    {
        T& input_momentum = *(input_cache[0]);
        input_momentum = ( m_mu * input_momentum ) - m_alpha * ( m_normalize_grad * gradient );
        input += input_momentum;
    }

    virtual const float& get_learning_rate() final { return m_alpha; }
    virtual void set_learning_rate( const float new_rate ) final { m_alpha = new_rate; }
    virtual size_t get_cache_size() final { return 1; }

private:

    float m_alpha;  // learning rate
    float m_lambda; // weight decay
    float m_mu;     //momentum
};

/* RMSprop solver implementation */
template <typename operatorF>
class solver_rmsprop : public solver_base
{
public:
    solver_rmsprop( const float alpha, const float mu )
    	: m_mu( mu ), m_alpha( alpha ), m_eps( 1e-8f ) {}
    solver_rmsprop( std::initializer_list<float> params_list )
    	: m_mu( 0.99f ), m_alpha( 0.0001f ), m_eps( 1e-8f )
    {
        if ( params_list.size() == 2 )
    	{
        	m_alpha     = params_list.begin()[0];
        	m_mu        = params_list.begin()[1];
        }
        else
            LOGGER(warning) << "solver_rmsprop::solver_rmsprop - invalid parameters number, keeping defaults" << std::endl;
    }
    solver_rmsprop() : m_mu( 0.99f ), m_alpha( 0.0001f ), m_eps( 1e-8f )
    {
        m_parameters_set = t_parameters_map( // assignment workaround added for clang/OSX
        	{ {"lr",std::ref(m_alpha)}, {"m",std::ref(m_mu)} }
        );
    }
    virtual ~solver_rmsprop() {}

    template<typename T>
    void update( T& input, T** input_cache, const T& gradient )
    {
        T& input_momentum = *(input_cache[0]);
        input_momentum = m_mu * input_momentum
            + ( 1 - m_mu ) * m_normalize_grad * m_normalize_grad * gradient * gradient;
        input -= m_alpha * m_normalize_grad * gradient / operatorF::sqrt( input_momentum + m_eps );
    }

    template<typename T>
    void update_redux( T& input, T** input_cache, const T& gradient ) // TODO : usefull???
    {
        update( input, input_cache, gradient );
    }

    virtual const float& get_learning_rate() final { return m_alpha; }
    virtual void set_learning_rate( const float new_rate ) final { m_alpha = new_rate; }
    virtual size_t get_cache_size() final { return 1; }

private:

    float m_mu;         // decay term
    float m_alpha;      // learning rate
    const float m_eps;  // constant value to avoid zero-division
};

/* Adadelta solver implementation */
template <typename operatorF>
class solver_adadelta : public solver_base
{
public:
    solver_adadelta( const float alpha, const float mu )
    	: m_mu( mu ), m_alpha( alpha ), m_eps( 1e-8f ) {}
    solver_adadelta( std::initializer_list<float> params_list )
    	: m_mu( 0.99f ), m_alpha( 0.0001f ), m_eps( 1e-8f )
    {
        if ( params_list.size() == 2 )
    	{
        	m_alpha     = params_list.begin()[0];
        	m_mu        = params_list.begin()[1];
        }
        else
            LOGGER(warning) << "solver_adadelta::solver_adadelta - invalid parameters number, keeping defaults" << std::endl;
    }
    solver_adadelta() : m_mu( 0.99f ), m_alpha( 0.0001f ), m_eps( 1e-8f )
    {
        m_parameters_set = t_parameters_map( // assignment workaround added for clang/OSX
        	{ {"lr",std::ref(m_alpha)}, {"m",std::ref(m_mu)} }
        );
    }
    virtual ~solver_adadelta() {}

    template<typename T>
    void update( T& input, T** input_cache, const T& gradient )
    {
        T& input_momentum1 = *(input_cache[0]);
        T& input_momentum2 = *(input_cache[1]);
        input_momentum1 = m_mu * input_momentum1 + ( 1 - m_mu ) * gradient * gradient;
        T dx = operatorF::sqrt( ( input_momentum2 + m_eps ) / ( input_momentum1 + m_eps ) ) * gradient;
        input_momentum2 = m_mu * input_momentum2 + ( 1 - m_mu ) * dx * dx;
        input -= m_alpha * dx;
    }

    template<typename T>
    void update_redux( T& input, T** input_cache, const T& gradient ) // TODO : usefull???
    {
        update( input, input_cache, gradient );
    }

    virtual const float& get_learning_rate() final { return m_alpha; }
    virtual void set_learning_rate( const float new_rate ) final { m_alpha = new_rate; }
    virtual size_t get_cache_size() final { return 2; }

private:

    float m_mu;         // decay term
    float m_alpha;      // learning rate
    const float m_eps;  // constant value to avoid zero-division
};

} /*namespace neurocl*/

#endif //SOLVER_H
