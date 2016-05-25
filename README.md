# neurocl
::: Neural network C++ implementations :::

Neurocl (**Neuro** **C**omputing **L**ibrary) was meant to experiment various 'from scratch' implementations of neural network layer schemes, focusing on matrix expression of feed forwarding/backward propagating.
It only implements Fully Connected Neural Network (FCNN) scheme for now, but my plan is to have a Convolutional Neural Network (CNN) scheme ready as soon as possible.
There are two different FCNN implementations in Neurocl : one using standard Boost.ublas containers, and another one based on VexCL containers.  

The upcoming evolutions will also focus on optimizations dedicated to running NN based algorithms on small low power devices, like the ARM based Raspberry Pi.

I've been experimenting Neurocl on three main recognition/classification topics : mnist handwritten caracters recognition, automatic license plate recognition, and face recognition.

## Prerequisite:

main library dependencies:

- [Boost C++](https://github.com/boostorg)
- [VexCL](https://github.com/ddemidov/vexcl)
- [CImg](https://github.com/dtschump/CImg)

sample applications dependencies:

- [ccv](http://libccv.org) - modern computer vision library
- [picoPi2](https://github.com/ch3ll0v3k/picoPi2) - simple TTS API

## Building:

building ccv (used for face detection):

```shell
$ git clone https://github.com/liuliu/ccv.git
$ cd ccv/lib
$ ./configure
$ make
```

building picoPi2 (used for Text-To-Speech on the Pi) and create an invokable TTS bash script:

```shell
$ git clone https://github.com/ch3ll0v3k/picoPi2.git
$ cd picoPi2/lib
$ make
$ cd ../tts
$ ./configure.py --make --2wav
$ echo -e '#!/bin/sh\nexport LD_LIBRARY_PATH="../lib"\n./picoPi2Wav "$1" -o speech.wav\naplay speech.wav' > speak.sh
$ sh speak.sh "This is a test"
```

building neurocl (mainly header-only dependencies):

```shell
$ sudo apt-get install libboost-all-dev
$ git clone https://github.com/dtschump/CImg.git
$ git clone https://github.com/ddemidov/vexcl.git
$ git clone https://github.com/blackccpie/neurocl.git
$ cd neurocl
$ sh build_gcc.sh
```

## User Guide:

neurocl requires three main input files:

1. the topology description file: this is a structured text file describing the neural net layers.

    ```text
    layer:0:28x28
    layer:1:6x6
    layer:2:10x1
    ```

2. the neural net weights file: this is a binary file containing the layers weight and bias values. This file is managed internally by neurocl, but user has to specify the name of the weights file to load for training/classifying.

3. the training set description file: this is a structured text file containing a list of image sample locations along with their expected output vectors. This file is only useful for net training steps.

    ```text
    /home/my_user/train-data/sample1.bmp 0 0 0 0 0 1 0 0 0 0
    /home/my_user/train-data/sample2.bmp 1 0 0 0 0 0 0 0 0 0
    /home/my_user/train-data/sample3.bmp 0 0 0 0 1 0 0 0 0 0
    /home/my_user/train-data/sample4.bmp 0 1 0 0 0 0 0 0 0 0
    /home/my_user/train-data/sample5.bmp 0 0 0 0 0 0 0 0 0 1
    ...
    ```

4. neurocl has basic xml configuration support: if present in the runtime directory, the _neurocl.xml_ file will be parsed, but for now only one key parameter is manager, which is the neural network learning rate. The _neurocl.xml_ file is formatted as shown below:

    ```xml
    <neurocl>
	    <learning_rate>1.5</learning_rate>
    </neurocl>
    ```

neurocl main entry point is class **network_manager**:
- the network implementation backend is chosen at construction

    ```
    neurocl::network_manager net_manager(neurocl::network_manager::NEURAL_IMPL_BNU );
    ```

- for now there are 3 backends available:

    * **NEURAL_IMPL_BNU_REF** : the reference implementation only using boost::numeric::ublas containers and operators
    * **NEURAL_IMPL_BNU_FAST** : fast  implementation using boost::numeric::ublas containers but custom simd (neon/sse4) optimized operators
    * **NEURAL_IMPL_VEXCL** : _experimental_ vexcl reference implementation.


- a given network can be loaded, given its topology and weights file names

    ```c++
    net_manager.load_network( "topology.txt", "weights.bin" );
    ```
    
- once a network is loaded, it can be trained, or used for direct output computation

    ```c++
    neurocl::sample sample(...);
    net_manager.compute_output( sample );
    ```

## Targeted platforms:

- Mac OSX
- Debian Jessie
- Raspberry Pi Raspbian

## References:
- [Neural networks and deep learning](http://neuralnetworksanddeeplearning.com)
- [Stanford class about sparse auto-encoders](http://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf)

## ToRead:
- http://page.mi.fu-berlin.de/rojas/neural/chapter/K7.pdf
-> §7.3.3 concerning matrix form

## License:

neurocl source is distributed under [MIT License](https://en.wikipedia.org/wiki/MIT_License)
