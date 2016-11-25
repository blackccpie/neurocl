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

#ifndef LEARNING_SCHEDULER_H
#define LEARNING_SCHEDULER_H

#include <memory>

namespace neurocl { namespace convnet {

class solver_base;

class learning_scheduler
{
public:
    static learning_scheduler& instance() { static learning_scheduler ls; return ls; }

    void enable_scheduling( const bool enable );
    void push_error( const float error );
    void set_learning_rate( const float rate );
    const float& get_learning_rate();

protected:

    friend class solver_base;
    void register_solver( const std::shared_ptr<solver_base>& solver );

private:

    learning_scheduler();
    virtual ~learning_scheduler();

    void _assert_solver();

private:

    bool m_enabled;
    float m_cached_rate;
    float m_cached_error;
    size_t m_err_count;
    const size_t m_err_window;

    std::shared_ptr<solver_base> m_solver;
};

} /*namespace neurocl*/ } /*namespace convnet*/

#endif //LEARNING_SCHEDULER_H
