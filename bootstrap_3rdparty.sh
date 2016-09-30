#!/bin/sh

CIMG_DIR="CImg"
VEXCL_DIR="vexcl"
CCV_DIR="ccv"

pushd ..

# bootstrap Boost
sudo apt-get install libboost-all-dev

# bootstrap CImg
if [ -d "$CIMG_DIR" ]; then
    cd $CIMG_DIR; git pull; cd -
else
    git clone https://github.com/dtschump/CImg.git
fi

# bootstrap VexCL
if [ -d "$VEXCL_DIR" ]; then
    cd $VEXCL_DIR
    git pull
    cd -
else
    git clone https://github.com/ddemidov/vexcl.git
fi

# bootstrap ccv
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
