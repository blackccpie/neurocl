#!/bin/bash

command_exists() {
    type "$1" &> /dev/null ;
}

if command_exists curl ; then
    curl http://blackccpie.free.fr/nets/alpr.zip > alpr.zip
    unzip -o alpr.zip; rm alpr.zip

    curl http://blackccpie.free.fr/nets/mnist.zip > mnist.zip
    unzip -o mnist.zip; rm mnist.zip
else
    wget http://blackccpie.free.fr/nets/alpr.zip
    unzip -o alpr.zip; rm alpr.zip

    wget http://blackccpie.free.fr/nets/mnist.zip
    unzip -o mnist.zip; rm mnist.zip
fi
