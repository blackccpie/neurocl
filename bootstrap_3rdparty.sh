#!/bin/bash

if [ "$(uname)" == "Darwin" ]; then
    echo "Sorry but this script does not support OSX yet!"
    exit 1
fi

if [ ! "$BASH_VERSION" ] ; then
    echo "Please do not use sh to run this script but bash instead!"
    exit 1
fi

CIMG_DIR="CImg"
VEXCL_DIR="vexcl"
CCV_DIR="ccv"

pushd ..

# bootstrap Boost
echo "--> boostraping Boost"
sudo apt-get install -y libboost-all-dev

# bootstrap OpenCL
echo "--> boostraping OpenCL"
sudo aptitude install -y opencl-dev

# bootstrap CMake
echo "--> boostraping CMake"
sudo apt-get install -y cmake

# bootstrap CImg
echo "--> boostraping CImg"
if [ -d "$CIMG_DIR" ]; then
    cd $CIMG_DIR; git pull; cd -
else
    git clone https://github.com/dtschump/CImg.git
fi

# bootstrap VexCL
echo "--> boostraping VexCL"
if [ -d "$VEXCL_DIR" ]; then
    cd $VEXCL_DIR
    git pull
    cd -
else
    git clone https://github.com/ddemidov/vexcl.git
fi

# bootstrap ccv
echo "--> boostraping ccv"
if [ -d "$CCV_DIR" ]; then
    cd $CCV_DIR
    git pull
    cd lib
    ./configure
    make
    cd -
else
    git clone https://github.com/liuliu/ccv.git
    cd ccv/lib
    ./configure
    make
    cd -
fi

popd
