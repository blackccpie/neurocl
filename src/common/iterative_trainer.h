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

#ifndef ITERATIVE_TRAINER_H
#define ITERATIVE_TRAINER_H

#include "interfaces/network_manager_interface.h"

namespace neurocl {

// used for custom external training
class iterative_trainer
{
public:
    iterative_trainer( std::shared_ptr<network_manager_interface> net_manager, const size_t batch_size )
        : m_net_manager( net_manager ), m_batch_pos( 0 ), m_batch_size( batch_size )
    {
        m_net_manager->set_training( true, {} );
        m_net_manager->prepare_training_epoch();
    }
    virtual ~iterative_trainer()
    {
        m_net_manager->finalize_training_epoch();
        m_net_manager->save_network();
        m_net_manager->set_training( false, {} );
    }

    void train_new( const sample& sample )
    {
        m_net_manager->train( sample, {} );

        ++m_batch_pos;

        if ( m_batch_pos >= m_batch_size )
        {
            m_net_manager->finalize_training_epoch();
            m_net_manager->prepare_training_epoch();
            m_batch_pos = 0;
        }
    }

private:

    size_t m_batch_pos;
    size_t m_batch_size;

    std::shared_ptr<network_manager_interface> m_net_manager;
};

} /*namespace neurocl*/

#endif //ITERATIVE_TRAINER_H
