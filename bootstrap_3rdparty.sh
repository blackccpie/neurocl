#!/bin/bash

if [ "$(uname -n)" == "raspberry" ]; then
    echo "Running bootstrap script on a Raspberry Pi!"
elif [ "$(uname -s)" == "Darwin" ]; then
    echo "Sorry but this script does not support OSX yet!"
    exit 1
else
    echo "Sorry but this script does not support this OS yet!"
    exit 1
fi

if [ ! "$BASH_VERSION" ] ; then
    echo "Please do not use sh to run this script but bash instead!"
    exit 1
fi

CIMG_DIR="CImg"
VEXCL_DIR="vexcl"
CCV_DIR="ccv"
BOOST_DIR="boost_1_58_0"
CMAKE_DIR="cmake-3.6.2"

pushd ..

#bootstrap git
echo "--> bootstrapping git"
sudo apt-get install -y git

#bootstrap libpython
echo "--> bootstrapping libpython"
sudo apt-get install -y libpython-dev

#bootstrap libbz2
echo "--> bootstrapping libbz2"
sudo apt-get install -y libbz2-dev

# bootstrap libjpeg
echo "--> bootstrapping libjpeg"
sudo apt-get install -y libjpeg-dev

# bootstrap Boost
echo "--> bootstrapping Boost"
#sudo apt-get install -y libboost-all-dev
if [ ! -d "$BOOST_DIR" ]; then
    cd ../Downloads
    wget http://sourceforge.net/projects/boost/files/boost/1.58.0/boost_1_58_0.tar.gz
    cd -
    tar -zxf ../Downloads/boost_1_58_0.tar.gz
    cd $BOOST_DIR
    ./bootstrap.sh
    ./b2 cxxflags="-std=c++11"
    sudo ./b2 cxxflags="-std=c++11" install
    cd -
fi

# bootstrap OpenCL
echo "--> bootstrapping OpenCL"
sudo aptitude install -y opencl-dev

# bootstrap CMake
echo "--> bootstrapping CMake"
#sudo apt-get install -y cmake
if [ ! -d "$CMAKE_DIR" ]; then
    cd ../Downloads
    wget https://cmake.org/files/v3.6/cmake-3.6.2.tar.gz
    cd -
    tar -zxf ../Downloads/cmake-3.6.2.tar.gz
    cd $CMAKE_DIR
    ./configure
    make
    sudo make install
fi

# bootstrap CImg
echo "--> bootstrapping CImg"
if [ -d "$CIMG_DIR" ]; then
    cd $CIMG_DIR; git pull; cd -
else
    git clone https://github.com/dtschump/CImg.git
fi

# bootstrap VexCL
echo "--> bootstrapping VexCL"
if [ -d "$VEXCL_DIR" ]; then
    cd $VEXCL_DIR
    git pull
    cd -
else
    git clone https://github.com/ddemidov/vexcl.git
fi

# bootstrap ccv
echo "--> bootstrapping ccv"
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
