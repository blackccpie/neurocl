#!/bin/bash

command_exists() {
    type "$1" &> /dev/null ;
}

if command_exists curl ; then
    curl http://blackccpie.free.fr/nets/alpr.tgz > alpr.tgz
    tar -zxf alpr.tgz; rm alpr.tgz

    curl http://blackccpie.free.fr/nets/mnist.tgz > mnist.tgz
    tar -zxf mnist.tgz; rm mnist.tgz
else
    wget http://blackccpie.free.fr/nets/alpr.tgz
    tar -zxf alpr.tgz; rm alpr.tgz

    wget http://blackccpie.free.fr/nets/mnist.tgz
    tar -zxf mnist.tgz; rm mnist.tgz
fi
