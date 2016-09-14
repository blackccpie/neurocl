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

#include "speech_manager.h"

#include <boost/lockfree/spsc_queue.hpp>
#include <boost/thread.hpp>

#define BRAIN_QUEUE_SIZE 15

class thebrain
{
public:
    thebrain();
    virtual ~thebrain();

    void push_face_type( int type );

private:

    void _run();

private:

	bool m_bStop;

	size_t m_current_compute_range;
    int m_current_face_type;

    speech_manager m_speech_manager;
    std::shared_ptr<boost::thread> m_thread;

    boost::lockfree::spsc_queue<int, boost::lockfree::capacity<BRAIN_QUEUE_SIZE> > m_queue;
};
