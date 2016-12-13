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

#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <atomic>
#include <thread>
#include <mutex>
#include <array>
#include <list>
#include <functional>
#include <condition_variable>

namespace neurocl {

/**
 *  Simple thread_pool that creates `thread_count` threads upon its creation,
 *  and pulls from a queue to get new jobs.
 *
 *  This class requires a number of c++11 features be present in your compiler.
 */

class thread_pool
{
public:

    thread_pool( const size_t thread_count )
        : m_jobs_left( 0 )
        , m_bailout( false )
        , m_finished( false )
        , m_thread_count( thread_count )
    {
        for( unsigned i = 0; i < m_thread_count; ++i )
            m_threads.emplace_back( std::thread( [this,i]{ this->task(); } ) );
    }

    /**
     *  JoinAll on deconstruction
     */
    ~thread_pool() {
        join_all();
    }

    /**
     *  Get the number of threads in this pool
     */
    inline unsigned size() const {
        return m_thread_count;
    }

    /**
     *  Get the number of jobs left in the queue.
     */
    inline unsigned jobs_remaining() {
        std::lock_guard<std::mutex> guard( m_queue_mutex );
        return m_queue.size();
    }

    /**
     *  Add a new job to the pool. If there are no jobs in the queue,
     *  a thread is woken up to take the job. If all threads are busy,
     *  the job is added to the end of the queue.
     */
    void add_job( std::function<void(void)> job ) {
        std::lock_guard<std::mutex> guard( m_queue_mutex );
        m_queue.emplace_back( job );
        ++m_jobs_left;
        m_job_available_var.notify_one();
    }

    /**
     *  Join with all threads. Block until all threads have completed.
     *  Params: WaitForAll: If true, will wait for the queue to empty
     *          before joining with threads. If false, will complete
     *          current jobs, then inform the threads to exit.
     *  The queue will be empty after this call, and the threads will
     *  be done. After invoking `ThreadPool::JoinAll`, the pool can no
     *  longer be used. If you need the pool to exist past completion
     *  of jobs, look to use `ThreadPool::WaitAll`.
     */
    void join_all( bool wait_for_all = true ) {
        if( !m_finished ) {
            if( wait_for_all ) {
                wait_all();
            }

            // note that we're done, and wake up any thread that's
            // waiting for a new job
            m_bailout = true;
            m_job_available_var.notify_all();

            for( auto &x : m_threads )
                if( x.joinable() )
                    x.join();
            m_finished = true;
        }
    }

    /**
     *  Wait for the pool to empty before continuing.
     *  This does not call `std::thread::join`, it only waits until
     *  all jobs have finshed executing.
     */
    void wait_all() {
        if( m_jobs_left > 0 ) {
            std::unique_lock<std::mutex> lk( m_wait_mutex );
            m_wait_var.wait( lk, [this]{ return this->m_jobs_left == 0; } );
            lk.unlock();
        }
    }

private:

    /**
     *  Take the next job in the queue and run it.
     *  Notify the main thread that a job has completed.
     */
    void task() {
        while( !m_bailout ) {
            next_job()();
            --m_jobs_left;
            m_wait_var.notify_one();
        }
    }

    /**
     *  Get the next job; pop the first item in the queue,
     *  otherwise wait for a signal from the main thread.
     */
    std::function<void(void)> next_job() {
        std::function<void(void)> res;
        std::unique_lock<std::mutex> job_lock( m_queue_mutex );

        // Wait for a job if we don't have any.
        m_job_available_var.wait( job_lock, [this]() ->bool { return m_queue.size() || m_bailout; } );

        // Get job from the queue
        if( !m_bailout ) {
            res = m_queue.front();
            m_queue.pop_front();
        }
        else { // If we're bailing out, 'inject' a job into the queue to keep jobs_left accurate.
            res = []{};
            ++m_jobs_left;
        }
        return res;
    }

private:

    size_t m_thread_count;

    std::vector<std::thread> m_threads;
    std::list<std::function<void(void)>> m_queue;

    std::atomic_int         m_jobs_left;
    std::atomic_bool        m_bailout;
    std::atomic_bool        m_finished;
    std::condition_variable m_job_available_var;
    std::condition_variable m_wait_var;
    std::mutex              m_wait_mutex;
    std::mutex              m_queue_mutex;
};

} // namespace neurocl

#endif //THREAD_POOL_H
