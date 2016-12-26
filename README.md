# neurocl
::: Neural network C++ implementations :::

Neurocl (**Neuro** **C**omputing **L**ibrary) was meant to experiment various 'from scratch' implementations of neural network layer schemes, focusing on matrix expression of _feed forward_/_backward propagation_ steps.
Initial release _v0.1_ only implemented Multi-Layer Perceptron (*MLP*) scheme, whereas _v0.2_ now incorporates a working Convolutional Neural Network (*CONVNET*) scheme.

There are two different **_MLP_** implementations in Neurocl : one using standard _Boost.ublas_ containers, and another one based on _VexCL_ containers.

There is only one **_CONVNET_** implementation for now, based on a tensor abstraction class, using _Boost.ublas_ containers.

The upcoming evolutions will also focus on optimizations dedicated to running neural network based algorithms on small low power devices, like the ARM based Raspberry Pi.

I've been experimenting Neurocl on three main recognition/classification topics : mnist handwritten caracters recognition, automatic license plate recognition, and face recognition.

## Prerequisites:

main library dependencies:

- [Boost C++ (1.58 minimum)](https://github.com/boostorg)
- [VexCL](https://github.com/ddemidov/vexcl)
- [CImg](https://github.com/dtschump/CImg)

sample applications dependencies:

- [ccv](http://libccv.org) - modern computer vision library
- [picoPi2](https://github.com/ch3ll0v3k/picoPi2) - simple TTS API

NOTE : there is a 3rd party software bootstrap script available for Raspbian:

```shell
$ sh bootstrap_3rdparty.sh
```

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
$ sh build_gcc_xxx.sh
```

## User Guide:

neurocl requires three main input files:

1. the **topology description** file: this is a structured text file describing the neural net layers.

    ```text
    layer:in:0:28x28x1
    layer:conv:1:24x24x3:5
    layer:pool:2:12x12x3
    layer:full:3:100x1x1
    layer:out:4:10x1:1
    ```

2. the **neural net weights** file: this is a binary file containing the layers weight and bias values. This file is managed internally by neurocl, but user has to specify the name of the weights file to load for training/classifying.

3. the **training set description** file: this is a structured text file containing a list of image sample locations along with their expected output vectors. This file is only useful for net training steps.

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
        <scheme>CONVNET</scheme>
        <backend>UBLAS</backend>
	    <learning_rate>1.5</learning_rate>
    </neurocl>
    ```

neurocl main entry point is interface **network_manager_interface**, which can only be returned with the help of the factory class **network_factory**:
- the network scheme can be built according to the xml configuration:

    ```
    std::shared_ptr<network_manager_interface> net_manager = network_factory::build();
    ```

    or given a specific scheme:

    ```
    std::shared_ptr<network_manager_interface> net_manager =
        network_factory::build( network_factory::t_neural_impl::NEURAL_IMPL_CONVNET );
    ```

    The two availables schemes are:
    - **NEURAL_IMPL_MLP**,
    - **NEURAL_IMPL_CONVNET**

    There are 3 MLP backends available:

        * *NEURAL_IMPL_BNU_REF* : the reference implementation only using boost::numeric::ublas containers and operators
        * *NEURAL_IMPL_BNU_FAST* : _experimental_ fast  implementation using boost::numeric::ublas containers but custom simd (neon/sse4) optimized operators (for now layer sizes should be multiples of 4)
        * *NEURAL_IMPL_VEXCL* : _experimental_ vexcl reference implementation.

        But for now it is default hardcoded to *NEURAL_IMPL_BNU_REF*

    For now there is a unique CONVNET backend.

- a given network can be loaded, given its topology and weights file names

    ```c++
    net_manager->load_network( "topology.txt", "weights.bin" );
    ```

- once a network is loaded, it can be trained with the help of the *samples_manager* class:

    ```c++
    samples_manager smp_train_manager;
    smp_train_manager.load_samples( "../nets/mnist/training/mnist-train.txt" );

    net_manager->batch_train( smp_train_manager, NB_EPOCHS, BATCH_SIZE );
    ```


- or used for direct output computation:

    ```c++
    neurocl::sample sample(...);
    net_manager->compute_output( sample );
    ```

## Architecture:

MLP Class diagram:

![PlantUML class diagram](http://plantuml.com/plantuml/png/XP91Ri8m44NtFiM8Rbf4KBfYWxBa0XmWcfXnOc4co3QXKilTGIMnQYSDAqlVppFpvriQT0wO_BMrnrAsh7GDsosyxrTlkzrca-SVu3JNXdpBK1J2jpNvXYny2ysUh499uNrGX8pgLdmfAtIftD6NDE8s0LjI4wf2vnFvX8mrsKHLsb3P81_CwChXwMo6GVHZNFIwez9Rr1pW9-H2TP4QWVNwfvYm7Jbx1VL6OooiAZN-0kj7XSNd0fPPzdl-ttgEZfOtCfvbHV9T4jCpmD3rBzBdCKeYWeOSvgbdAH19UXFC7G00)

## Targeted platforms:

- Mac OSX
- Debian Jessie
- Ubuntu 16.04 LTS
- Raspberry Pi Raspbian

## References:
- [Neural networks and deep learning](http://neuralnetworksanddeeplearning.com)
- [Stanford CS class CS231n](http://cs231n.github.io)
- [Stanford class about sparse auto-encoders](http://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf)

## ToRead:
- http://page.mi.fu-berlin.de/rojas/neural/chapter/K7.pdf
-> ß7.3.3 concerning matrix form

## License:

neurocl source is distributed under [MIT License](https://en.wikipedia.org/wiki/MIT_License)
