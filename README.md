# neurocl

::: Neural network C++ implementations :::

[![License](https://img.shields.io/github/license/mashape/apistatus.svg)](https://raw.githubusercontent.com/blackccpie/neurocl/master/LICENSE)
-----------------

![Training graph](http://blackccpie.free.fr/nets/neurocl-web.png)

Neurocl (**Neuro** **C**omputing **L**ibrary) was meant to experiment various 'from scratch' implementations of neural network layer schemes, focusing on matrix expression of _feed forward_/_backward propagation_ steps.
Initial release _v0.1_ only implemented Multi-Layer Perceptron (*MLP*) scheme, whereas _v0.2_ now incorporates a working Convolutional Neural Network (*CONVNET*) scheme.

There are two different **_MLP_** implementations in Neurocl : one using standard _Boost.ublas_ containers, and another one based on _VexCL_ containers.

There is only one **_CONVNET_** implementation for now, based on a tensor abstraction class, using _Boost.ublas_ containers.

As MLP was developped first, and emphasis was put on the CONVNET implementation for the past months, things are still managed a bit differently between these two architectures, but my plan is to factorize all possible things (solver, activation, loss function etc..) to have a common framework between them.

Please note that CONVNET implementation is still missing some configuration features and does not benefit of a fast tensor container implementation yet.

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

NOTE: if experiencing problems due to *libccv* linking issues, rebuild the latter with the *-fPIC* compiler directive.

## User Guide:

### File management

neurocl requires three main input files:

1. the **topology description** file: this is a structured text file describing the neural net layers.

    FOR **MLP**:

    ```text
    layer:0:28x28
    layer:1:6x6
    layer:2:10x1
    ```

    *NOTE : MLP does not allow configurable activation functions for now, the default is sigmoid activations for all layers.*

    FOR **CONVNET**:

    ```text
    layer:in:0:28x28x1
    layer:conv:1:24x24x3:5
    layer:drop:2:24x24x3
    layer:pool:3:12x12x3
    layer:full:4:100x1x1
    layer:out:5:10x1:1
    ```

    *NOTE : CONVNET does not allow configurable activation functions for now, the default configuration is ReLU for convolutional layers, and Softmax with cross entropy error for the output layer. It can be edited in the __src/convnet/network.cpp__ file.*

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

4. neurocl has basic xml configuration support: if present in the runtime directory, the _neurocl.xml_ file will be parsed, but for now only a limited set of parameters are managed. The _neurocl.xml_ file is formatted as shown below:

    Example of MLP configuration file (default SGD solver):

    ```xml
    <neurocl>
    	<implementation>MLP</implementation>
		<learning_rate>1.5</learning_rate>
    </neurocl>
    ```

    Example of CONVNET configuration file with a RMSPROP solver:

    ```xml
    <neurocl>
    	<implementation>CONVNET</implementation>
    	<solver type="RMSPROP" lr="0.001" m="0.99"/>
    </neurocl>
    ```

### Training and using the network

neurocl main entry point is interface **network_manager_interface**, which can only be returned with the help of the factory class **network_factory**:
- the network scheme can be built according to the xml configuration:

    ```c++
    std::shared_ptr<network_manager_interface> net_manager = network_factory::build();
    ```

	or given a specific scheme:

    ```c++
    std::shared_ptr<network_manager_interface> net_manager =
        network_factory::build( network_factory::t_neural_impl::NEURAL_IMPL_CONVNET );
    ```

    The two availables schemes are:

    * **NEURAL_IMPL_MLP**

        3 backends available:
        * *NEURAL_IMPL_BNU_REF* : the reference implementation only using boost::numeric::ublas containers and operators.
        * *NEURAL_IMPL_BNU_FAST* : _experimental_ fast  implementation using boost::numeric::ublas containers but custom simd (neon/sse4) optimized operators (for now layer sizes should be multiples of 4).
        * *NEURAL_IMPL_VEXCL* : _experimental_ vexcl reference implementation.
    * **NEURAL_IMPL_CONVNET**

        2 backends available:
        * *CONVNET* : the single-threaded implementation.
        * *CONVNET_PARALLEL* : the multi-threaded implementation (only for training for now).

	_**Note1**_ : MLP default backend is hardcoded to *NEURAL_IMPL_BNU_REF*.

	_**Note2**_ : CONVNET default backend is hardcoded to *CONVNET*.

	_**Note3**_ : these hardcoded settings can be changed in the *network_factory* class.

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

- The reference sample application to look for best practice code is __*mnist_autotrainer*__, located in the *apps* directory.

## Visualizing training data

The __*mnist_autotrainer*__ application also illustrates dumping training data in a dedicated _.csv_ file. This data file can be graphically represented in a web page using [CanvasJS](http://canvasjs.com) framework (as shown at the top of this doc), ie by running a lightweight python webserver and viewing the right url:

```shell
$ cd neurocl
$ python -m SimpleHTTPServer
$ firefox http://127.0.0.1:8000/web
```

## Architecture:

*NOTE : for the sake of clarity, the following diagrams have been simplified to the main classes. More detailed diagrams will be added as appendices soon.*

MLP Class diagram:

![PlantUML class diagram](http://www.plantuml.com/plantuml/png/ZP3H2i8m34NV-nLblWaR-0C--HybgwaEqsoadGhglrkA22iLbvT0xWcvDmmUmmGMDUCiewNEqwGtXrpwePGb2269yJRAnGmSKLp2JS8ApGs4vWny99H2yi1mGibBXTJpR7e8M8olUBTKGGLTPUeWHAKgrtIpQ_IsRNEl6bsxj4nUkjTTzzp_ONEaDU0dvnBz_gdC_GDJhpS0)

CONVNET Class diagram:

![PlantUML class diagram](http://www.plantuml.com/plantuml/png/bLBBJiCm4BpxAwnog8He9Ewe9wvmucvPxgQDs7gjjHE4eFzE7iPnJ2gLN2BFpCwilMlr6RpJ1gI5bibWPSXs5eAyUnu_-IN4Dj0HmWcuFsmDQkzb0Ek9boUbJMWw7Hgolc0yOGUbd1pmgXh9UrMtraJo8aHZ0t0zz8dwitSIKy7Gh1gHfpdyRH_P0vEZDao2YDKjulMAnhf2xuXm0VI6IPDRF00pSUyKTVqRVX2_gxEs-WYiuhiMRDqWg9Es1qm1lrIwydffeNQQDIAW45U2vUPn8-ztk_6_6l-Vmh_DHKPMrVV-1baoVTt4aPOtkT5frYVBbzJdi3XfMfFX59PFCSVdzSZHUSroT5wD75qiNKw8omy0)

## Targeted platforms:

- Mac OSX
- Debian Jessie
- Ubuntu 16.04 LTS
- Raspberry Pi Raspbian

## References:

MLP:
- [Neural networks and deep learning](http://neuralnetworksanddeeplearning.com)
- [Stanford class about sparse auto-encoders](http://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf)

CONVNETS:
- [Stanford CS class CS231n](http://cs231n.github.io)

CLASSIFIERS:
- [How to implement a neural network](http://peterroelants.github.io/posts/neural_network_implementation_part01)
- [The Softmax function and its derivative](http://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative)

SOLVERS:
- [Caffe solver tutorial](http://caffe.berkeleyvision.org/tutorial/solver.html)

FRENCH SPEAKING:

- http://clement.chatelain.free.fr/enseignements/rdn.pdf
- http://emilie.caillault.free.fr/doc/EPoisson2001.pdf

## ToRead:
- http://page.mi.fu-berlin.de/rojas/neural/chapter/K7.pdf
-> ß7.3.3 concerning matrix form

## Inspirations
- [tiny-dnn](http://github.com/tiny-dnn)
- [ConvNetJS](http://github.com/karpathy/convnetjs)

## License:

neurocl source is distributed under [MIT License](https://en.wikipedia.org/wiki/MIT_License)
